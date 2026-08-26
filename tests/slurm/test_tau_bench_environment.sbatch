#!/bin/bash
#SBATCH --job-name=test-tau-environment
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# Exercise the pinned Tau v1 reward on every retail training task before a GPU
# recipe or its completed-rollout replay path is admitted.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"

readonly TAU_COMMIT=09c26a85efd1d65168cfb57865ca2ca278c8153d
readonly TAU_SOURCE_DIR="${WS}/src/tau-bench"
[[ -d "${TAU_SOURCE_DIR}/.git" ]] || { echo "missing Tau checkout: ${TAU_SOURCE_DIR}" >&2; exit 1; }
[[ "$(git -C "${TAU_SOURCE_DIR}" rev-parse HEAD)" == "${TAU_COMMIT}" ]] || {
    echo "Tau checkout is not pinned to ${TAU_COMMIT}" >&2
    exit 1
}

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${TAU_SOURCE_DIR}:/root/tau-bench" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/tau-bench:/root/miles
export WANDB_MODE=offline

python3 -m pytest -q tests/fast/experiments/test_tau_bench.py

TAU_TRAIN_SCRIPT=/root/miles/experiments/scripts/tau_bench/async/nemotron3-agentic-retail/qwen3-4b-instruct-2507/train.sh
if grep -q GEMINI_API_KEY "${TAU_TRAIN_SCRIPT}"; then
    echo "Tau train.sh must not serialize the Gemini key in Ray submit arguments or runtime metadata" >&2
    exit 1
fi

python3 - <<"PY"
import asyncio
import json
import os

from experiments.src.environments.tau_bench.compat import install_litellm_import_stub
from experiments.src.environments.tau_bench.task_identity import TAU_COMMIT, _task_digest, _task_dict

install_litellm_import_stub()
from tau_bench.envs import get_env

path = "/data/tau-bench/tau1-retail-train-miles.jsonl"
with open(path, encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
assert len(rows) == 500, len(rows)
verified_path = "/data/tau-bench/tau1-retail-train-reward-verified-miles.jsonl"
with open(verified_path, encoding="utf-8") as stream:
    verified_rows = [json.loads(line) for line in stream if line.strip()]

env = get_env(
    env_name="retail",
    user_strategy="human",
    user_model="offline",
    task_split="train",
    task_index=0,
)
nondiscriminating = []
for row_number, row in enumerate(rows, start=1):
    metadata = row["metadata"]
    task_index = int(metadata["tau_task_index"])
    task = env.tasks[task_index]
    assert metadata["tau_commit"] == TAU_COMMIT
    assert metadata["tau_task_sha256"] == _task_digest(_task_dict(task))

    env.task_index = task_index
    env.task = task
    env.data = env.data_load_func()
    env.actions = []
    if float(env.calculate_reward().reward) != 0.0:
        nondiscriminating.append(task_index)

    env.data = env.data_load_func()
    env.actions = []
    for action in task.actions:
        env.step(action)
    assert float(env.calculate_reward().reward) == 1.0, (row_number, task_index, "ground-truth")

assert nondiscriminating, "expected the pinned Tau v1 reward defect to remain visible"
verified_indices = {int(row["metadata"]["tau_task_index"]) for row in verified_rows}
assert verified_indices == set(range(len(rows))) - set(nondiscriminating)
assert all(row["metadata"].get("tau_reward_verified") is True for row in verified_rows)
print(
    f"Official Tau retail reward: rows={len(rows)} verified={len(verified_rows)} "
    f"ground-truth=1 no-op-positive={len(nondiscriminating)} indices={nondiscriminating}"
)

from transformers import AutoTokenizer

from experiments.src.environments.tau_bench.generator import (
    _build_action,
    _load_environment,
    _tokenize_user_observation,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser, tokenize_tool_responses

tokenizer = AutoTokenizer.from_pretrained("/ckpt/hf/Qwen3-4B-Instruct-2507", trust_remote_code=True)
verified_metadata = verified_rows[0]["metadata"]
verified_env = _load_environment(verified_metadata)
initial_messages = [
    {"role": "system", "content": str(verified_env.wiki)},
    {"role": "user", "content": "I need help with an order."},
]
initial_tokens = tokenizer.apply_chat_template(
    initial_messages,
    tokenize=True,
    add_generation_prompt=True,
    tools=verified_env.tools_info,
    return_dict=False,
)
assert initial_tokens and isinstance(initial_tokens[0], int)
user_tokens = _tokenize_user_observation(tokenizer, "Yes, please proceed.")
assert user_tokens and isinstance(user_tokens[0], int)
tool_tokens = tokenize_tool_responses(
    [
        {
            "role": "tool",
            "name": "get_order_details",
            "tool_call_id": "tau_preflight",
            "content": "{\"status\":\"pending\"}",
        }
    ],
    tokenizer,
)
assert tool_tokens
parser = create_tool_call_parser(verified_env.tools_info, "qwen25")
action = _build_action(
    parser,
    "<tool_call>\n{\"name\":\"get_order_details\",\"arguments\":{\"order_id\":\"#W0000000\"}}\n</tool_call>",
)
assert action.name == "get_order_details"
assert action.kwargs == {"order_id": "#W0000000"}
print(
    f"Tau Qwen3 tool/chat preflight: initial={len(initial_tokens)} "
    f"user-observation={len(user_tokens)} append-only "
    f"tool-observation={len(tool_tokens)} parser=ok"
)

from experiments.src.environments.tau_bench.user_simulator import (
    DEFAULT_GEMINI_MODEL,
    GeminiRequestError,
    STOP_MARKER,
    build_gemini_payload,
    generate_gemini_user,
    parse_gemini_response,
    require_gemini_api_key,
)

gemini_payload = build_gemini_payload(
    [
        {"role": "system", "content": "private task"},
        {"role": "user", "content": "agent greeting"},
        {"role": "assistant", "content": "customer request"},
    ],
    max_output_tokens=512,
    temperature=0.7,
    top_p=0.95,
    seed=42,
)
assert [content["role"] for content in gemini_payload["contents"]] == ["user", "model"]
assert gemini_payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
parsed = parse_gemini_response(
    {
        "candidates": [
            {
                "content": {"parts": [{"text": f"Finished {STOP_MARKER}"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"candidatesTokenCount": 3},
    }
)
assert parsed.text == STOP_MARKER
assert parsed.output_tokens == 3

try:
    require_gemini_api_key({})
except GeminiRequestError:
    pass
else:
    raise AssertionError("Gemini must fail closed without a process-environment key")


async def check_gemini_transport():
    secret = "slurm-unit-secret"
    os.environ["GEMINI_API_KEY"] = secret
    calls = []

    async def fake_post(url, payload, headers):
        calls.append((url, payload, headers))
        if len(calls) == 1:
            raise GeminiRequestError("rate limited", status_code=429)
        return {"candidates": [{"content": {"parts": [{"text": "Retry worked."}]}}]}

    result = await generate_gemini_user(
        [
            {"role": "system", "content": "private task"},
            {"role": "user", "content": "How can I help?"},
        ],
        post_json=fake_post,
        model=DEFAULT_GEMINI_MODEL,
        max_retries=1,
        retry_backoff=0,
    )
    assert result.text == "Retry worked."
    assert len(calls) == 2
    url, request_payload, headers = calls[-1]
    assert secret not in url
    assert secret not in json.dumps(request_payload)
    assert headers["x-goog-api-key"] == secret
    del os.environ["GEMINI_API_KEY"]


asyncio.run(check_gemini_transport())
print("Tau Gemini user simulator: payload, parsing, retry, fail-closed, secret handling=ok")
PY

# Ray documents that variables present when the cluster starts remain visible
# to Jobs drivers and workers even when the submitted runtime_env adds other
# variables. Verify that contract without placing the sentinel in the command or
# persisted Jobs runtime metadata.
export MILES_TAU_RAY_SECRET_SENTINEL="tau-ray-inherit-${SLURM_JOB_ID}"
RAY_PORT=$((20000 + SLURM_JOB_ID % 10000))
RAY_DASHBOARD_PORT=$((40000 + SLURM_JOB_ID % 10000))
RAY_TEST_TEMP="${SLURM_TMPDIR:-/tmp}/tau-ray-env-${SLURM_JOB_ID}"
mkdir -p "${RAY_TEST_TEMP}"
cleanup_ray_test() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray_test EXIT INT TERM
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address=127.0.0.1 --port "${RAY_PORT}" \
    --num-cpus 4 --disable-usage-stats --temp-dir "${RAY_TEST_TEMP}" \
    --dashboard-host=127.0.0.1 --dashboard-port "${RAY_DASHBOARD_PORT}"

RAY_TEST_SUBMISSION_ID="tau-ray-env-${SLURM_JOB_ID}"
RAY_TEST_RUNTIME_ENV="{\"env_vars\":{\"WANDB_MODE\":\"offline\",\"TAU_RAY_PROBE_RUNTIME_ENV\":\"1\"}}"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --submission-id "${RAY_TEST_SUBMISSION_ID}" \
    --runtime-env-json="${RAY_TEST_RUNTIME_ENV}" \
    -- python3 /root/miles/tests/slurm/support/tau_ray_env_probe.py

python3 - "http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${RAY_TEST_SUBMISSION_ID}" <<"PY"
import json
import os
import sys

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(sys.argv[1])
info = client.get_job_info(sys.argv[2])
persisted = json.dumps(
    {
        "entrypoint": info.entrypoint,
        "metadata": info.metadata,
        "runtime_env": info.runtime_env,
    },
    sort_keys=True,
)
sentinel = os.environ["MILES_TAU_RAY_SECRET_SENTINEL"]
assert sentinel not in persisted
assert "MILES_TAU_RAY_SECRET_SENTINEL" not in persisted
assert "GEMINI_API_KEY" not in persisted
print("Ray Jobs metadata secret scan: entrypoint=clean runtime_env=clean metadata=clean")
PY

cleanup_ray_test
trap - EXIT INT TERM
'
