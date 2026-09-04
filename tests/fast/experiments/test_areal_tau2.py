from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.datasets.areal_tau2.prepare import _schedule_summary, adapt_areal_tau2
from experiments.src.environments.areal_tau2.generator import generate
from experiments.src.environments.areal_tau2.runtime import (
    AReaLTau2Session,
    _generate_user_message,
    _user_message_from_assistant,
    _validate_metadata,
    canonical_digest,
)
from experiments.src.environments.tau_bench.continuation import (
    TAU_CONTINUATION_KEY,
    message_history_digest,
)
from experiments.src.environments.tau_bench.generator import _generate_tau
from experiments.src.environments.tau_bench.runtime import TauReset, TauStep
from experiments.src.environments.tau_bench.task_identity import TAU_COMMIT, TAU_PACKAGE_VERSION
from experiments.src.protocols.areal_tau2 import (
    AREAL_TAU2_DATASET,
    AREAL_TAU2_DB_SHA256,
    AREAL_TAU2_EXPECTED_COUNTS,
    AREAL_TAU2_EXPECTED_ROWS,
    AREAL_TAU2_INTERACTION_MODE,
    AREAL_TAU2_POLICY,
    AREAL_TAU2_REVISION,
    AREAL_TAU2_VERIFIER,
)
from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

REPO_ROOT = Path(__file__).resolve().parents[3]


def _task_data(task_id: str = "duplicate-id") -> dict:
    return {
        "id": task_id,
        "description": {"purpose": "test"},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": "airline",
                "reason_for_call": "change a flight",
                "known_info": "reservation ABC123",
                "unknown_info": None,
                "task_instructions": "Ask the agent to change the flight.",
            },
        },
        "ticket": None,
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [],
            "env_assertions": None,
            "communicate_info": [],
            "nl_assertions": None,
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "issues": None,
        "required_documents": None,
        "user_tools": None,
    }


def _source_row(task_id: str = "duplicate-id") -> dict:
    return {
        "id": task_id,
        "description": {"purpose": "test"},
        "user_scenario": _task_data(task_id)["user_scenario"],
        "evaluation_criteria": '{"actions":[],"communicate_info":[]}',
        "db_path": "tau2_rl_database/tau2_airline_db.json",
    }


def test_areal_tau2_contract_is_exactly_the_rl_split() -> None:
    assert AREAL_TAU2_EXPECTED_COUNTS == {"airline": 1148, "retail": 563, "telecom": 271}
    assert AREAL_TAU2_EXPECTED_ROWS == 1982
    assert len(AREAL_TAU2_DB_SHA256) == 9
    assert AREAL_TAU2_INTERACTION_MODE == "stateful_multi_turn_user_simulator_environment"


def test_adapter_uses_row_index_because_source_task_ids_are_not_unique(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_data = _task_data()
    monkeypatch.setattr(
        "experiments.src.datasets.areal_tau2.prepare._task_data",
        lambda _row: (task_data, "airline"),
    )

    def resolve_state(*_args, **_kwargs):
        return "agent-hash", None, ["gold-action:ToolError"]

    first = adapt_areal_tau2(
        _source_row(),
        row_index=1,
        dataset_root=tmp_path,
        state_resolver=resolve_state,
    )
    second = adapt_areal_tau2(
        _source_row(),
        row_index=2,
        dataset_root=tmp_path,
        state_resolver=resolve_state,
    )

    assert first["metadata"]["source_row_id"] == "airline:0001:duplicate-id"
    assert second["metadata"]["source_row_id"] == "airline:0002:duplicate-id"
    assert first["metadata"]["interaction_mode"] == AREAL_TAU2_INTERACTION_MODE
    assert first["metadata"]["stateful_environment"] is True
    assert first["metadata"]["user_simulator"] is True
    assert first["metadata"]["tau_expected_agent_db_hash"] == "agent-hash"
    assert first["metadata"]["tau_expected_user_db_hash"] is None
    assert first["metadata"]["tau_gold_replay_errors"] == ["gold-action:ToolError"]


def test_schedule_summary_counts_optimizer_updates_not_trajectories() -> None:
    summary = _schedule_summary(1982)["updates_by_rollout_batch_size_and_epoch"]
    assert summary["32"] == {"5": 310, "6": 372}
    assert summary["63"] == {"5": 158, "6": 189}
    assert summary["192"] == {"5": 52, "6": 62}


def test_training_recipe_caps_epochs_and_uses_inflight_replay() -> None:
    recipe = (
        REPO_ROOT
        / "experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953"
    )
    run_script = (recipe / "run.sbatch").read_text(encoding="utf-8")
    train_script = (recipe / "train.sh").read_text(encoding="utf-8")

    assert ': "${TRAIN_EPOCHS:=6}"' in run_script
    assert '[[ "${TRAIN_EPOCHS}" =~ ^[1-6]$ ]]' in run_script
    assert '#SBATCH --nodes=8' in run_script
    assert ': "${TOTAL_NODES:=8}"' in run_script
    assert ': "${ROLLOUT_BATCH_SIZE:=63}"' in run_script
    assert ': "${N_SAMPLES_PER_PROMPT:=16}"' in run_script
    assert "AREAL_TAU2_ROWS=1982" in run_script
    assert "AREAL_TAU2_ROWS * TRAIN_EPOCHS" in run_script
    assert ': "${USE_REPLAY_BUFFER:=1}"' in run_script
    assert ': "${REPLAY_BUFFER_TYPE:=inflight}"' in run_script
    assert ': "${ALLOW_TAU_REPLAY_ABLATION:=0}"' in run_script
    assert 'if [[ "${ALLOW_TAU_REPLAY_ABLATION}" == 0 ]]; then' in run_script
    assert '[[ "${EVAL_INTERVAL}" == 0 ]]' in run_script
    assert ': "${TAU_LOG_LEVEL:=ERROR}"' in run_script
    assert ': "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL:=0}"' in run_script
    assert ': "${WANDB_PROJECT:=async-rl-tau}"' in run_script
    assert 'MODEL_NAME:=Qwen3-4B-Agentic-SFT-953' in run_script
    assert 'HF_MODEL_NAME:=Qwen3-4B-Base/${AGENTIC_SFT_RUN}/iter_0000953' in run_script
    assert 'MODEL_PROFILE:=qwen3-4B' in run_script
    assert "nemotron-sft-agentic-v2-qwen3-preserve-thinking" in run_script
    assert "Qwen3-4B-Instruct-2507" not in run_script
    assert 'export LOGURU_LEVEL="${TAU_LOG_LEVEL}"' in train_script
    assert '--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE}"' in train_script
    assert '--log-replay-resume-metrics' in train_script
    assert '--custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"' in train_script
    assert '[[ "${TAU_LOG_OVERHEAD}" == 0 ]] ||' in train_script
    assert "--tau-log-overhead" in train_script
    assert '[[ "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 0 ]] ||' in train_script
    assert "--tau-overlap-db-restore-with-prefill" in train_script
    assert '[[ -z "${REF_LOAD}" ]] || CKPT_ARGS+=' in train_script


def test_40k_staleness_study_is_a_twelve_arm_self_extending_chain() -> None:
    recipe_root = REPO_ROOT / "experiments/scripts/tau_bench/async/areal-tau2"
    launcher_path = recipe_root / "submit_staleness_truncation_sweep.sh"
    segment_path = recipe_root / "internal/run_staleness_truncation_segment.sbatch"
    launcher = launcher_path.read_text(encoding="utf-8")
    segment = segment_path.read_text(encoding="utf-8")
    train_script = (
        recipe_root / "qwen3-4b-agentic-sft-953/train.sh"
    ).read_text(encoding="utf-8")

    for fixed_setting in (
        "readonly TARGET_UPDATES=180",
        "readonly MAX_CONTEXT_LEN=40960",
        "readonly MAX_RESPONSE_LEN=40960",
        "readonly -a STALENESS_LEVELS=(8 16 20)",
        "readonly -a RATIOS=(1:7 2:6)",
        "readonly -a TRUNCATION_MODES=(zero-reward zero-loss)",
        '"LR=1e-6"',
        '"IS_CORRECTION=tis"',
        '"USE_REPLAY_BUFFER=1"',
        '"REPLAY_BUFFER_TYPE=inflight"',
        '"TAU_OVERLAP_DB_RESTORE_WITH_PREFILL=1"',
        '"TAU_LOG_OVERHEAD=1"',
        '"SAVE_INTERVAL=10"',
        '"HF_SAVE_INTERVAL=10"',
        '"SAMPLE_STALENESS_MAX_BIN=32"',
    ):
        assert fixed_setting in launcher

    assert "#SBATCH --time=04:00:00" in segment
    assert "#SBATCH --signal=B:USR1@900" in segment
    assert '--dependency="afterany:${SLURM_JOB_ID}"' in segment
    assert "trap request_checkpoint_and_successor USR1" in segment
    assert "refusing a potentially duplicate submission" in launcher
    assert 'ACTIVE_JOB_IDS["${run_name}"]' in launcher
    assert '--save-trigger-sentinel "${SAVE_TRIGGER_SENTINEL}"' in train_script
    assert '--zero-loss-on-truncated' in train_script

    environment = {
        **os.environ,
        "RUN_NAMESPACE": "pytest-study",
        "WANDB_MODE": "offline",
    }
    result = subprocess.run(
        ["bash", str(launcher_path)],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.stdout.count("tau40k-s") == 12
    assert "effective_epochs=5.721493" in result.stdout
    assert "M=20 is an effectively unbounded control" in result.stdout

    for script_path in (launcher_path, segment_path):
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def _metadata() -> dict:
    task = _task_data()
    db_path = "tau2_rl_database/tau2_airline_db.json"
    return {
        "source": "areal-tau2-rl",
        "verifier": AREAL_TAU2_VERIFIER,
        "interaction_mode": AREAL_TAU2_INTERACTION_MODE,
        "environment_policy": AREAL_TAU2_POLICY,
        "stateful_environment": True,
        "user_simulator": True,
        "eval_only": False,
        "dataset_repo": AREAL_TAU2_DATASET,
        "dataset_revision": AREAL_TAU2_REVISION,
        "source_row_id": "airline:0001:duplicate-id",
        "tau_domain": "airline",
        "tau_task": task,
        "tau_task_sha256": canonical_digest(task),
        "tau_db_path": db_path,
        "tau_db_sha256": AREAL_TAU2_DB_SHA256[db_path],
        "tau_expected_agent_db_hash": "agent-hash",
        "tau_expected_user_db_hash": None,
        "tau_package_version": TAU_PACKAGE_VERSION,
        "tau_commit": TAU_COMMIT,
    }


def test_runtime_metadata_fails_closed() -> None:
    metadata = _metadata()
    _validate_metadata(metadata)
    with pytest.raises(ValueError, match="task digest"):
        _validate_metadata({**metadata, "tau_task_sha256": "wrong"})
    with pytest.raises(ValueError, match="DB identity"):
        _validate_metadata({**metadata, "tau_db_sha256": "wrong"})
    broken = dict(metadata)
    del broken["tau_expected_agent_db_hash"]
    with pytest.raises(ValueError, match="missing"):
        _validate_metadata(broken)


def test_user_simulator_preserves_tool_call_only_response_atomically() -> None:
    from tau2.data_model.message import AssistantMessage, ToolCall

    assistant_message = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                name="lookup_account",
                arguments={"account_id": "123"},
                requestor="assistant",
            )
        ],
    )

    user_message = _user_message_from_assistant(assistant_message)

    assert user_message.content is None
    assert len(user_message.tool_calls) == 1
    assert user_message.tool_calls[0].id == "call-1"
    assert user_message.tool_calls[0].name == "lookup_account"
    assert user_message.tool_calls[0].arguments == {"account_id": "123"}
    assert user_message.tool_calls[0].requestor == "user"


def test_user_simulator_retries_empty_responses_with_deterministic_seeds() -> None:
    calls = []
    responses = iter(
        (
            SimpleNamespace(content=None, tool_calls=None),
            SimpleNamespace(content="  ", tool_calls=[]),
            SimpleNamespace(
                content="I found it.",
                tool_calls=None,
                cost=0.0,
                usage=None,
                raw_data=None,
            ),
        )
    )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return next(responses)

    user_message = _generate_user_message(
        fake_generate,
        model="openai/user-model",
        messages=["history"],
        tools=None,
        llm_args={"num_retries": 4, "seed": 42, "temperature": 0.7},
    )

    assert user_message.content == "I found it."
    assert [call["seed"] for call in calls] == [42, 43, 44]
    assert all(call["num_retries"] == 4 for call in calls)
    assert all(call["call_name"] == "user_simulator_response" for call in calls)


def test_areal_generator_uses_dedicated_session(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_generate(input, *, session_factory):
        captured["input"] = input
        captured["session_factory"] = session_factory
        return "generated"

    monkeypatch.setattr("experiments.src.environments.areal_tau2.generator._generate_tau", fake_generate)
    input = SimpleNamespace(sample=SimpleNamespace(metadata={"verifier": AREAL_TAU2_VERIFIER}))

    assert asyncio.run(generate(input)) == "generated"
    assert captured == {"input": input, "session_factory": AReaLTau2Session}

    rejected = SimpleNamespace(sample=SimpleNamespace(metadata={"verifier": "tau3_environment"}))
    with pytest.raises(ValueError, match="rejects verifier"):
        asyncio.run(generate(rejected))


def test_areal_generator_declares_inflight_replay_capability() -> None:
    assert generate.supports_inflight_replay is True


def test_tau_generator_resumes_policy_prefix_and_environment_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_kwargs):
            text = "".join(f"<{message['role']}>{message.get('content') or ''}" for message in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(text.encode()) if tokenize else text

        def decode(self, token_ids):
            return bytes(token_ids).decode()

    resume_request_started = threading.Event()

    class FakeSession:
        instances = []

        def __init__(self, metadata, _user, *, max_steps, continuation):
            assert metadata["verifier"] == AREAL_TAU2_VERIFIER
            assert max_steps == 200
            self.continuation = continuation
            self.actions = []
            self.instances.append(self)

        def reset(self):
            if self.continuation is not None:
                assert resume_request_started.wait(timeout=2.0)
            return TauReset(
                system_prompt="policy",
                messages=[{"role": "user", "content": "start"}],
                tools=[],
            )

        def snapshot(self):
            history = [{"role": "user", "content": "start", "timestamp": "discarded"}]
            return {
                "message_history": history,
                "message_history_sha256": message_history_digest(history),
                "agent_db_hash": "agent-state",
                "user_db_hash": "user-state",
                "orchestrator_step_count": 1,
                "orchestrator_num_errors": 0,
            }

        def step(self, action):
            self.actions.append(action)
            return TauStep(
                observations=[],
                reward=1.0,
                terminated=True,
                truncated=False,
                reward_info={"reward": 1.0},
                simulation_run={"messages": [action]},
            )

        def close(self):
            return None

    class FakeParser:
        def parse_non_stream(self, response):
            return response, []

    outputs = iter(
        (
            {
                "text": "hel",
                "meta_info": {
                    "finish_reason": {"type": "abort"},
                    "weight_version": "5",
                    "response_weight_version_segments": [[0, 3, 5]],
                },
            },
            {
                "text": "lo",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "weight_version": "6",
                    "response_weight_version_segments": [[0, 2, 6]],
                },
            },
        )
    )
    prefill_tokens = []

    def fake_payload(_args, token_ids, _sampling_params):
        prefill_tokens.append(list(token_ids))
        return {"input_ids": list(token_ids)}, None

    post_calls = 0

    async def fake_post(*_args, **_kwargs):
        nonlocal post_calls
        post_calls += 1
        if post_calls == 2:
            resume_request_started.set()
        return next(outputs)

    async def fake_update(_args, sample, _payload, output, *, update_loss_mask):
        assert update_loss_mask is True
        token_ids = list(output["text"].encode())
        sample.tokens.extend(token_ids)
        sample.response += output["text"]
        sample.response_length += len(token_ids)
        sample.loss_mask.extend([1] * len(token_ids))
        sample.rollout_log_probs.extend([0.0] * len(token_ids))
        sample.update_policy_version_from_meta_info(output["meta_info"])
        finish_type = output["meta_info"]["finish_reason"]["type"]
        sample.status = Sample.Status.ABORTED if finish_type == "abort" else Sample.Status.COMPLETED

    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.create_tool_call_parser",
        lambda *_args, **_kwargs: FakeParser(),
    )
    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.compute_request_payload",
        fake_payload,
    )
    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.compute_routing_headers",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr("experiments.src.environments.tau_bench.generator.post", fake_post)
    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.update_sample_from_response",
        fake_update,
    )
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_response_len=64,
        rollout_seed=7,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        tau_log_overhead=False,
        tau_overlap_db_restore_with_prefill=True,
        tau_max_steps=200,
        tau_max_turns=4,
        tau_tool_call_parser="fake",
        tau_user_max_retries=1,
        tau_user_max_tokens=32,
        tau_user_model="fake-user",
        tau_user_provider="nvidia",
        tau_user_request_timeout=1.0,
        tau_user_temperature=0.0,
        tau_user_top_p=1.0,
    )
    state = SimpleNamespace(args=args, tokenizer=FakeTokenizer(), aborted=False)
    original = Sample(index=3, metadata={"verifier": AREAL_TAU2_VERIFIER})

    interrupted = asyncio.run(
        _generate_tau(
            GenerateFnInput(state=state, sample=original, sampling_params={}, evaluation=False),
            session_factory=FakeSession,
        )
    ).samples
    continuation = interrupted.metadata[TAU_CONTINUATION_KEY]
    interrupted_tokens = list(interrupted.tokens)

    assert interrupted.status == Sample.Status.ABORTED
    assert interrupted.response == "hel"
    assert continuation["policy_response_prefix"] == "hel"
    assert continuation["policy_prefix_response_tokens"] == 3
    assert continuation["agent_db_hash"] == "agent-state"
    assert interrupted.response_weight_version_segments == [[[0, 3, 5]]]

    resumed = asyncio.run(
        _generate_tau(
            GenerateFnInput(state=state, sample=interrupted, sampling_params={}, evaluation=False),
            session_factory=FakeSession,
        )
    ).samples

    assert FakeSession.instances[0].continuation is None
    assert FakeSession.instances[1].continuation == continuation
    assert resume_request_started.is_set()
    assert prefill_tokens[1][: len(interrupted_tokens)] == interrupted_tokens
    assert FakeSession.instances[1].actions == ["hello"]
    assert resumed.status == Sample.Status.COMPLETED
    assert resumed.reward == 1.0
    assert resumed.response == "hello"
    assert resumed.weight_versions == ["5", "6"]
    assert resumed.response_weight_version_segments == [[[0, 3, 5]], [[0, 2, 6]]]
    assert resumed.metadata["tau_done"] is True
    assert TAU_CONTINUATION_KEY not in resumed.metadata
