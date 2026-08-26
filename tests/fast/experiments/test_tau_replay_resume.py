from __future__ import annotations

import asyncio
import copy
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.reward_sets import tau
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.replay_buffer import load_replay_buffer, save_replay_buffer
from miles.rollout.replay_buffer_codec import (
    SAMPLE_CODEC_STATE_KEY,
    ReplayBufferSampleEncoder,
)
from miles.utils import arguments as arguments_module
from miles.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[3]
TAU_RECIPE = (
    REPO_ROOT
    / "experiments/scripts/tau_bench/async/nemotron3-agentic-retail/"
    "qwen3-4b-instruct-2507"
)
TAU_REWARD_PATH = "experiments.src.reward_sets.tau.reward"
TAU_GENERATE_PATH = "experiments.src.environments.tau_bench.generator.generate"


def _tau_sample() -> Sample:
    return Sample(
        group_index=7,
        index=70,
        prompt=[{"role": "system", "content": "policy"}, {"role": "user", "content": "help"}],
        tokens=[10, 11, 12],
        response="<tool_call>...</tool_call><tool_response>ok</tool_response>",
        response_length=3,
        reward=1.0,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3],
        status=Sample.Status.COMPLETED,
        metadata={
            "source": "tau-bench-retail",
            "verifier": "tau_bench_environment",
            "tau_commit": "09c26a85efd1d65168cfb57865ca2ca278c8153d",
            "tau_task_index": 3,
            "tau_task_sha256": "digest",
            "tau_user_backend": "local-policy",
            "tau_user_model": "rollout-policy",
            "tau_turns": 2,
            "tau_done": True,
            "tau_reward_info": {"reward": 1.0, "details": {"database": "matched"}},
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "done"},
            ],
        },
    )


def _durable_state(sample: Sample) -> dict:
    encoder = ReplayBufferSampleEncoder()
    prompt = copy.deepcopy(sample)
    prompt.response = ""
    prompt.response_length = 0
    prompt.reward = None
    prompt.loss_mask = []
    prompt.rollout_log_probs = []
    prompt.status = Sample.Status.PENDING
    prompt_ref = encoder.encode_group([prompt])
    result_ref = encoder.encode_group([sample])
    return {
        "dataset_fingerprint": "tau-config-a",
        "replay_buffer_type": "rollout",
        "data_source": {
            "sample_offset": 17,
            "epoch_id": 2,
            "sample_group_index": 8,
            "sample_index": 71,
            "metadata": {"source": "tau"},
        },
        "applied_weight_version": 5,
        "pending_prompts": [prompt_ref],
        "ready_items": [
            {
                "prompt_group": prompt_ref,
                "result": result_ref,
                "submission_weight_version": 5,
            }
        ],
        "drain_progress": [],
        "prepared_batches": [],
        "regeneration_group_ids": [],
        "inflight_items": [],
        SAMPLE_CODEC_STATE_KEY: encoder.finish(),
    }


def test_completed_tau_trajectory_and_reward_round_trip_through_durable_rollout_buffer(tmp_path: Path):
    original = _tau_sample()
    save_replay_buffer(tmp_path, 4, _durable_state(original))
    restored = load_replay_buffer(tmp_path, 4, expected_fingerprint="tau-config-a")

    restored_sample = Sample.from_dict(restored["ready_items"][0]["result"][0])
    assert restored_sample.to_dict() == original.to_dict()
    assert restored["inflight_items"] == []
    assert restored["data_source"]["sample_offset"] == 17


def test_rollout_cursor_state_restores_exactly_without_transient_environment_state():
    source = object.__new__(RolloutDataSource)
    source.args = SimpleNamespace(rollout_global_dataset=False, rollout_shuffle=False)
    state = {
        "sample_offset": 17,
        "epoch_id": 2,
        "sample_group_index": 8,
        "sample_index": 71,
        "metadata": {"source": "tau"},
    }

    source.restore_checkpoint_state(state)

    assert source.checkpoint_state() == state
    assert not hasattr(source, "environment")


def test_tau_reward_scores_only_static_rows_and_rejects_environment_rows(monkeypatch):
    async def fake_static_reward(args, sample_or_samples, **kwargs):
        if isinstance(sample_or_samples, list):
            return [1.0] * len(sample_or_samples)
        return 1.0

    monkeypatch.setitem(tau._HANDLERS, "expert_action", fake_static_reward)
    static = Sample(metadata={"verifier": "expert_action"})
    environment = Sample(metadata={"verifier": "tau_bench_environment"}, reward=1.0)

    assert asyncio.run(tau.reward(None, static)) == 1.0
    with pytest.raises(ValueError, match="rejects verifier"):
        asyncio.run(tau.reward(None, environment))


@pytest.mark.asyncio
async def test_stateful_tau_reward_bypasses_the_static_custom_reward(monkeypatch):
    async def stateful_generate(input):
        input.sample.status = Sample.Status.COMPLETED
        input.sample.reward = 1.0
        return GenerateFnOutput(samples=input.sample)

    async def reject_reward_call(*_args, **_kwargs):
        raise AssertionError("the static Tau reward must not rescore an environment trajectory")

    monkeypatch.setattr(inference_rollout_common, "async_rm", reject_reward_call)
    state = SimpleNamespace(
        args=SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=False,
        ),
        generate_fn_semaphore=asyncio.Semaphore(1),
        aborted=False,
        generate_function=stateful_generate,
    )
    sample = Sample(metadata={"verifier": "tau_bench_environment"})

    restored = await inference_rollout_common.generate_and_rm(state, sample, {})

    assert restored is sample
    assert restored.reward == 1.0


def _replay_args(**overrides) -> Namespace:
    values = dict(
        use_replay_buffer=True,
        replay_buffer_type="rollout",
        fully_async=True,
        train_backend="megatron",
        rollout_global_dataset=True,
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        advantage_estimator="grpo",
        use_critic=False,
        custom_rm_path=TAU_REWARD_PATH,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rollout_data_postprocess_path=None,
        buffer_filter_path=None,
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path=None,
        load_debug_rollout_data=False,
        ci_inject_rollout_data_path=None,
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        lora_rank=0,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        use_indexer_replay=False,
        use_rollout_indexer_replay=False,
        update_weight_transfer_mode="broadcast",
        update_weights_interval=1,
        save="/tmp/checkpoint",
        save_interval=1,
        replay_buffer_keep_last=2,
        custom_generate_function_path=TAU_GENERATE_PATH,
    )
    values.update(overrides)
    return Namespace(**values)


def test_tau_rollout_replay_is_allowlisted_after_serialization_audit():
    assert TAU_REWARD_PATH in arguments_module._REPLAY_BUFFER_VALIDATED_CUSTOM_RM_PATHS
    args = _replay_args()

    arguments_module._validate_replay_buffer(args)


def test_tau_custom_generator_rejects_inflight_replay():
    args = _replay_args(
        replay_buffer_type="inflight",
    )

    with pytest.raises(ValueError, match="built-in single-turn generate function"):
        arguments_module._validate_replay_buffer(args)


def _resolve_identity(slurm_job_id: str, *, use_replay_buffer: str = "0") -> tuple[str, str, str]:
    run_script = (TAU_RECIPE / "run.sbatch").read_text(encoding="utf-8")
    task_family = next(
        line.removeprefix("TASK_FAMILY=")
        for line in run_script.splitlines()
        if line.startswith("TASK_FAMILY=")
    )
    dataset_tag = next(
        line.removeprefix("DATASET_TAG=")
        for line in run_script.splitlines()
        if line.startswith("DATASET_TAG=")
    )
    environment = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": slurm_job_id,
        "MODEL_NAME": "Qwen3-4B-Instruct-2507",
        "DATASET_TAG": dataset_tag,
        "TASK_FAMILY": task_family,
        "PLACEMENT": "async",
        "ADVANTAGE_ESTIMATOR": "grpo",
        "EPS_CLIP": "0.2",
        "EPS_CLIP_HIGH": "0.28",
        "EPS_CLIP_C": "",
        "RATIO_DENOMINATOR": "actor",
        "IS_CORRECTION": "tis",
        "TIS_CLIP": "2.0",
        "TIS_CLIP_LOW": "0",
        "MIS_PROFILE": "",
        "USE_OPSM": "0",
        "M2PO_BUDGET": "0.04",
        "OPSM_DELTA": "1e-4",
        "KL_LOSS_COEF": "0.00",
        "LR": "1e-6",
        "MAX_RESPONSE_LEN": "16384",
        "NUM_STEPS_PER_ROLLOUT": "1",
        "ROLLOUT_BATCH_SIZE": "192",
        "GLOBAL_BATCH_SIZE": "3072",
        "N_SAMPLES_PER_PROMPT": "16",
        "TRAIN_SEED": "1234",
        "ROLLOUT_SEED": "42",
        "QUEUE_TYPE": "queue-recycle",
        "QUEUE_FACTOR": "1",
        "MAX_WEIGHT_STALENESS": "4",
        "STALENESS_REFERENCE": "prefill",
        "ZERO_REWARD_ON_TRUNCATED": "1",
        "USE_REPLAY_BUFFER": use_replay_buffer,
        "REPLAY_BUFFER_TYPE": "rollout",
        "REPLAY_BUFFER_IDENTITY_TAG": "1",
        "CONFIG_TAG": (
            "4node-rollout-length-16k-lr1e-6-rbs192-gbs3072-n16-turns30-"
            "userlocal-policy-rollout-policy-tseed1234-rseed42"
        ),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n%s\\n" "$RUN_NAME" "$CKPT_PATH" "$CONFIG_TAG"',
            "bash",
            str(REPO_ROOT / "experiments/common/run_identity.sh"),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return tuple(result.stdout.splitlines())


def test_tau_recipe_uses_rollout_replay_semantics_and_stable_resume_identity():
    run_script = (TAU_RECIPE / "run.sbatch").read_text(encoding="utf-8")
    train_script = (TAU_RECIPE / "train.sh").read_text(encoding="utf-8")
    clean_script = (REPO_ROOT / "experiments/common/clean_checkpoint.sh").read_text(encoding="utf-8")
    train_async = (REPO_ROOT / "train_async.py").read_text(encoding="utf-8")

    assert ': "${USE_REPLAY_BUFFER:=0}"' in run_script
    assert ': "${REPLAY_BUFFER_TYPE:=rollout}"' in run_script
    assert "replay requires REPLAY_BUFFER_TYPE=rollout" in train_script
    assert TAU_REWARD_PATH in run_script and TAU_REWARD_PATH in train_script
    assert TAU_GENERATE_PATH in run_script and TAU_GENERATE_PATH in train_script
    assert ': "${CLEAN_CHECKPOINT:=0}"' in clean_script
    assert "CLEAN_CHECKPOINT=1" not in run_script

    first = _resolve_identity("100001")
    resumed = _resolve_identity("100002")
    assert first == resumed
    assert first[2].endswith("-zero-trunc-no-rb")

    replay_first = _resolve_identity("100003", use_replay_buffer="1")
    replay_resumed = _resolve_identity("100004", use_replay_buffer="1")
    assert replay_first == replay_resumed
    assert replay_first[2].endswith("-zero-trunc-rb-rollout")
    assert replay_first != first

    replay_publish = train_async.index("await rollout_manager.save.remote(rollout_id)")
    model_publish = train_async.index("await save_training_model(actor_model, rollout_id")
    commit = train_async.index("await rollout_manager.mark_replay_buffer_committed.remote(rollout_id)")
    assert replay_publish < model_publish < commit
