#!/bin/bash
#SBATCH --job-name=test-harbor-e2b-live
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# One explicit live lifecycle probe for Harbor's native E2B provider. This job
# never reads .env and never places the API key in argv, output, or task data.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_harbor_e2b_live.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || exit 2
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached the live E2B probe" >&2
    exit 2
}
: "${HARBOR_ROOT:?set HARBOR_ROOT to the pinned Harbor checkout}"
: "${E2B_API_KEY:?export E2B_API_KEY in the submitting process; .env is not read}"
: "${E2B_LIVE_SOURCE_IMAGE:?set an admitted linux/amd64 image@sha256 digest}"

[[ "${E2B_LIVE_SOURCE_IMAGE}" =~ ^.+@sha256:[0-9a-f]{64}$ ]] || {
    echo "E2B_LIVE_SOURCE_IMAGE must be an immutable OCI digest" >&2
    exit 1
}

readonly EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85
readonly HARBOR_PYTHON="${HARBOR_PYTHON:-${HARBOR_ROOT}/.venv/bin/python}"

[[ "$(git -C "${HARBOR_ROOT}" rev-parse HEAD)" == "${EXPECTED_HARBOR_COMMIT}" ]] || {
    echo "Harbor must be pinned to ${EXPECTED_HARBOR_COMMIT}" >&2
    exit 1
}
[[ -x "${HARBOR_PYTHON}" ]] || {
    echo "Missing Harbor Python; run 'uv sync --extra e2b' in HARBOR_ROOT" >&2
    exit 1
}

export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export HARBOR_E2B_REQUIRE_PREBUILT=true
export HARBOR_E2B_NO_NEW_PRIVS_USERS=1000
export PYTHONPATH="${REPO_ROOT}:${HARBOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/apply_harbor_e2b_overlays.sh" \
    "${HARBOR_ROOT}"

"${HARBOR_PYTHON}" - <<'PY'
import asyncio
import hashlib
import os
import secrets
import tempfile
from pathlib import Path

from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths


async def main() -> None:
    payload = b"harbor-e2b-live-lifecycle\n"
    source_image = os.environ["E2B_LIVE_SOURCE_IMAGE"]
    with tempfile.TemporaryDirectory(prefix="harbor-e2b-live-") as raw_dir:
        root = Path(raw_dir)
        environment_dir = root / "environment"
        environment_dir.mkdir()
        trial_paths = TrialPaths(trial_dir=root / "trial")
        trial_paths.mkdir()
        local_source = root / "upload.bin"
        local_target = root / "download.bin"
        local_source.write_bytes(payload)

        environment = E2BEnvironment(
            environment_dir=environment_dir,
            environment_name="miles-live-lifecycle",
            session_id="one-sandbox",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image=source_image),
            network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        )

        try:
            build = await environment._create_template(
                alias=f"miles-live-{secrets.token_hex(16)}",
                skip_cache=True,
            )
            assert isinstance(build.template_id, str) and build.template_id
            assert isinstance(build.build_id, str) and build.build_id
            environment._template_id = build.template_id
            await environment._create_sandbox()
            assert environment._sandbox is not None
            await environment.ensure_dirs(
                environment._mount_targets(writable_only=True)
            )

            result = await environment.exec(
                "grep -q '^NoNewPrivs:[[:space:]]*1$' /proc/self/status && "
                "grep -q '^CapEff:[[:space:]]*0000000000000000$' /proc/self/status "
                "&& printf 'native-e2b-ok'",
                user=1000,
            )
            assert result.return_code == 0
            assert result.stdout == "native-e2b-ok"

            network = await environment.exec(
                "if /usr/bin/timeout 5s /bin/bash --noprofile --norc -c "
                "'exec 3<>/dev/tcp/1.1.1.1/443'; then exit 41; "
                "else printf network-blocked; fi",
                user=1000,
            )
            assert network.return_code == 0
            assert network.stdout == "network-blocked"

            await environment.upload_file(local_source, "/tmp/miles-upload.bin")
            digest_result = await environment.exec("sha256sum /tmp/miles-upload.bin")
            assert digest_result.return_code == 0
            assert digest_result.stdout.split()[0] == hashlib.sha256(payload).hexdigest()
            await environment.download_file("/tmp/miles-upload.bin", local_target)
            assert local_target.read_bytes() == payload
        finally:
            # E2BEnvironment.stop() deliberately treats teardown as best-effort.
            # This live probe calls its native kill operation directly so a
            # failed remote teardown fails the job instead of being logged only.
            if environment._sandbox is not None:
                await environment._stop_sandbox()
                environment._sandbox = None

        assert environment._sandbox is None

    print(
        "Harbor native E2B live lifecycle passed: immutable template-ID create, "
        "NoNewPrivs/capability/no-network checks, exec/upload/download/kill"
    )


asyncio.run(main())
PY
