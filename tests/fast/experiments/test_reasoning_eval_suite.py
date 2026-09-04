from __future__ import annotations

import asyncio
import base64
import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.evaluators.livecodebench import decode_private_tests
from experiments.tools.reasoning_eval import suite as reasoning_suite
from experiments.tools.reasoning_eval.suite import _parser, _request_completion, _score_batch
from miles.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeResponse:
    def __init__(self, message: dict[str, object]) -> None:
        self.status = 200
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def text(self) -> str:
        return json.dumps({"choices": [{"message": self._message, "finish_reason": "stop"}]})


class _FakeSession:
    def __init__(self, message: dict[str, object]) -> None:
        self.message = message
        self.payload: dict[str, object] | None = None

    def post(self, _endpoint: str, *, json: dict[str, object]) -> _FakeResponse:
        self.payload = json
        return _FakeResponse(self.message)


def _request_with_fake_session(session: _FakeSession, *, enable_thinking: bool):
    return asyncio.run(
        _request_completion(
            session,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
            endpoint="http://local/v1/chat/completions",
            model="model",
            row={"prompt": [{"role": "user", "content": "question"}]},
            row_index=0,
            repeat=0,
            max_tokens=16384,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_retries=1,
            enable_thinking=enable_thinking,
        )
    )


def test_reasoning_suite_can_request_non_thinking_final_content() -> None:
    args = _parser().parse_args(
        [
            "generate",
            "--input",
            "input.jsonl",
            "--output",
            "output.jsonl",
            "--endpoint",
            "http://local",
            "--model",
            "model",
            "--repeats",
            "1",
            "--max-tokens",
            "16384",
        ]
    )
    session = _FakeSession({"content": "final answer", "reasoning": None})

    result = _request_with_fake_session(session, enable_thinking=args.enable_thinking)

    assert result["response"] == "final answer"
    assert session.payload is not None
    assert session.payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_reasoning_suite_classifies_reasoning_only_output_as_terminal() -> None:
    session = _FakeSession(
        {"content": None, "reasoning_content": "answer hidden in reasoning"}
    )

    result = _request_with_fake_session(session, enable_thinking=True)

    assert result["response"] == ""
    assert result["generation_status"] == "empty_final_content"
    assert result["reasoning_length"] == len("answer hidden in reasoning")


def test_reasoning_suite_resumes_partial_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "candidates.jsonl"
    input_path.write_text(
        json.dumps({"prompt": [{"role": "user", "content": "question"}]}) + "\n",
        encoding="utf-8",
    )
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    partial_path.write_text(
        json.dumps(
            {
                "row_index": 0,
                "repeat": 0,
                "response": "",
                "finish_reason": "length",
                "generation_status": "empty_final_content",
                "reasoning_length": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def fake_completion(*_args: object, **kwargs: object) -> dict[str, object]:
        return {
            "row_index": kwargs["row_index"],
            "repeat": kwargs["repeat"],
            "response": "answer",
            "finish_reason": "stop",
            "generation_status": "ok",
            "reasoning_length": 0,
        }

    monkeypatch.setattr(reasoning_suite, "_request_completion", fake_completion)
    args = SimpleNamespace(
        input=input_path,
        output=output_path,
        endpoint="http://local",
        model="model",
        repeats=2,
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        concurrency=1,
        request_timeout=1,
        max_retries=1,
        enable_thinking=True,
        limit=None,
    )

    asyncio.run(reasoning_suite.generate(args))

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [(record["repeat"], record["generation_status"]) for record in records] == [
        (0, "empty_final_content"),
        (1, "ok"),
    ]


@pytest.mark.parametrize("task", ("aime24", "aime25", "aime26", "math500"))
def test_math_benchmark_reasoning_suite_uses_math_verifier(task: str) -> None:
    samples = [
        Sample(response=r"Reasoning. Answer: \boxed{73}", label="73"),
        Sample(response=r"Reasoning. Answer: \boxed{72}", label="73"),
    ]

    assert asyncio.run(_score_batch(task, samples)) == [1.0, 0.0]


@pytest.mark.parametrize("task", ("gpqa_diamond", "gpqa_main", "gpqa_extended"))
def test_gpqa_benchmark_reasoning_suite_uses_gpqa_verifier(task: str) -> None:
    samples = [
        Sample(response="Final answer: B", label="B"),
        Sample(response="Final answer: A", label="B"),
    ]

    assert asyncio.run(_score_batch(task, samples)) == [1.0, 0.0]


def test_reasoning_eval_suite_routes_verified_artifacts_and_repeat_counts() -> None:
    scripts = REPO_ROOT / "experiments/scripts/reasoning_eval"
    runner = (scripts / "run-suite.sbatch").read_text(encoding="utf-8")
    aime_runner = (scripts / "run-evaluation.sbatch").read_text(encoding="utf-8")
    post_training = (scripts / "run-suite-after-training.sbatch").read_text(
        encoding="utf-8"
    )
    scorer = (scripts / "score-suite.sbatch").read_text(encoding="utf-8")

    for script in (runner, scorer):
        assert 'TASKS_TEXT="${TASKS_TEXT//,/ }"' in script
        assert 'AIME_REPEATS="${AIME_REPEATS:-16}"' in script
        assert 'MATH500_REPEATS="${MATH500_REPEATS:-4}"' in script
        assert 'IFBENCH_REPEATS="${IFBENCH_REPEATS:-8}"' in script
        assert 'input="/data/aime-${year}/aime-${year}.jsonl"' in script
        assert "input=/data/math-500/math-500.jsonl" in script
        assert 'input="/data/gpqa/gpqa-${split}-miles.jsonl"' in script

    assert 'ENABLE_THINKING="${ENABLE_THINKING:-true}"' in runner
    assert "#SBATCH --gres=gpu:8" in runner
    assert "#SBATCH --gpus-per-node" not in runner
    assert "Qwen3-4B-Instruct-2507" not in runner
    assert "Qwen3-4B-Base/LR2.0e-5" in runner
    assert "iter_0004000" in runner
    assert "aime24 aime25 aime26 math500 livecodebench" in runner
    assert 'VLLM_STARTUP_ATTEMPTS="${VLLM_STARTUP_ATTEMPTS:-3}"' in runner
    assert "for startup_attempt in" in runner
    assert 'stop_server' in runner
    assert "aime24 aime25 aime26 math500 livecodebench" in scorer
    assert "#SBATCH --gres=gpu:8" in post_training
    assert "#SBATCH --gpus-per-node" not in post_training
    assert "Qwen3-4B-Instruct-2507" not in aime_runner
    assert "Qwen3-4B-Base/LR2.0e-5" in aime_runner
    assert "iter_0004000" in aime_runner
    assert 'ARM_NAME="${ARM_NAME:-sft-step4000}"' in aime_runner
    assert 'host_input="${DATASET_DIR}${input#/data}"' in runner
    assert 'sha256sum "${host_input}"' in runner
    assert "evaluation-contract.env" in runner
    assert "artifact-manifest.sha256" in runner
    assert "candidates contain an unclassified empty final response" in scorer


def test_livecodebench_decoder_accepts_the_pinned_private_test_encoding() -> None:
    tests = [{"input": "1 2\n", "output": "3\n"}]
    payload = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()

    assert decode_private_tests(payload) == tests
    assert decode_private_tests(json.dumps(tests)) == tests


def test_ifbench_evaluator_uses_a_staged_repo_without_runtime_install(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "ifbench"
    deps = tmp_path / "ifbench-deps"
    repo.mkdir()
    deps.mkdir()
    (deps / "nltk_data").mkdir()
    (deps / ".miles-ifbench-test").touch()
    (repo / "evaluation_lib.py").write_text(
        """
class InputExample:
    def __init__(self, *, key, instruction_id_list, prompt, kwargs):
        self.key = key
        self.instruction_id_list = instruction_id_list
        self.prompt = prompt
        self.kwargs = kwargs


class Result:
    def __init__(self, followed):
        self.follow_all_instructions = followed


def test_instruction_following_strict(example, prompt_to_response):
    return Result(prompt_to_response[example.prompt] == "followed")
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "IFBENCH_DEPS_PATH": str(deps),
            "IFBENCH_REPO_PATH": str(repo),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "evaluation_lib.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Miles Test",
            "-c",
            "user.email=miles-test@example.invalid",
            "commit",
            "-qm",
            "test evaluator",
        ],
        check=True,
    )
    program = """
import os
import subprocess

from miles.rollout.rm_hub import ifbench

ifbench.PINNED_IFBENCH_COMMIT = subprocess.check_output(
    ["git", "-C", os.environ["IFBENCH_REPO_PATH"], "rev-parse", "HEAD"], text=True
).strip()
ifbench.PINNED_DEPS_MARKER = ".miles-ifbench-test"

metadata = {
    "instruction_id_list": ["test:instruction"],
    "kwargs": [{}],
    "prompt_text": "prompt",
    "record_id": "1",
}
assert ifbench.compute_ifbench_reward("followed", None, metadata) == 1.0
assert ifbench.compute_ifbench_reward("missed", None, metadata) == 0.0
"""

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
    )
