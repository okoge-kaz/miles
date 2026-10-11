import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.src.nemo_gym.runtime.serve import configuration

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("scenario,expected", [("success", 0), ("training_failure", 7), ("gym_failure", 1)])
def test_four_task_job_supervises_scoring_without_a_user_model(tmp_path, scenario, expected):
    common = tmp_path / "experiments/common"
    common.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "experiments/common/run_nemo_gym_training.sh", common / "run_nemo_gym_training.sh")
    (tmp_path / "experiments/env.sh").write_text("""export OUTPUT_DIR="${MILES_REPO}/outputs"
export MILES_JOB_ID_SHORT=test
export CONTAINER_MOUNTS=unused
getent() { printf '10.0.0.1 STREAM host\n'; }
_miles_build_container_command() { MILES_CONTAINER_COMMAND=(python3 "${MILES_REPO}/service.py" "$@"); }
""")
    (common / "run_training.sh").write_text("""set -euo pipefail
trap 'touch "${MILES_REPO}/training-stopped"; exit 143' TERM
[[ -s "${OUTPUT_DIR}/nemo_gym/test/endpoints.json" ]]
[[ -n "${NEMO_GYM_ENDPOINTS}" ]]
touch "${MILES_REPO}/training-started"
case "${SCENARIO}" in
    success) exit 0 ;;
    training_failure) exit 7 ;;
    gym_failure) while :; do sleep 0.05; done ;;
esac
""")
    (tmp_path / "service.py").write_text("""import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["MILES_REPO"])
(root / "service-args.json").write_text(json.dumps(sys.argv[1:]))
endpoint = root / "outputs/nemo_gym/test/endpoints.json"
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
try:
    endpoint.write_text("{}")
    while True:
        if os.environ["SCENARIO"] == "gym_failure" and (root / "training-started").exists():
            sys.exit(23)
        time.sleep(0.05)
finally:
    endpoint.unlink(missing_ok=True)
    (root / "service-stopped").touch()
""")
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "experiments/scripts/staleness_multi_env/math_code_mcqa_if/run.sh")],
        env={"PATH": os.environ["PATH"], "MILES_REPO": str(tmp_path), "SCENARIO": scenario},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    arguments = json.loads((tmp_path / "service-args.json").read_text())
    services = arguments[arguments.index("--services") + 1 :]
    assert set(services) == {"code_gen", "mcqa", "instruction_following"}
    config, urls = configuration("127.0.0.1", 30190, services, None, 200)
    assert set(urls) == set(services)
    assert "user_model" not in config and "judge_model" not in config
    assert (tmp_path / "training-started").exists()
    assert (tmp_path / "service-stopped").exists()
    assert not (tmp_path / "outputs/nemo_gym/test/endpoints.json").exists()
    if scenario == "gym_failure":
        assert (tmp_path / "training-stopped").exists()
