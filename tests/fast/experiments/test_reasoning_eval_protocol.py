from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tools.reasoning_eval.protocol import (
    derive_protocol_name,
    resolve_protocol_name,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_protocol_name_preserves_the_pinned_64_repeat_identity() -> None:
    assert derive_protocol_name(
        temperature="0.6",
        top_p="0.95",
        top_k=20,
        effective_repeats=64,
    ) == (
        "eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-instruct-"
        "t0.6-p0.95-k20-aime64-v1"
    )


def test_thinking_protocol_has_a_distinct_identity() -> None:
    assert "-thinking-" in derive_protocol_name(
        temperature="0.6",
        top_p="0.95",
        top_k=20,
        effective_repeats=64,
        enable_thinking=True,
    )


@pytest.mark.parametrize("effective_repeats", [1, 8, 64])
def test_protocol_name_is_derived_from_effective_repeats(effective_repeats: int) -> None:
    name = resolve_protocol_name(
        temperature="0.6",
        top_p="0.95",
        top_k=20,
        effective_repeats=effective_repeats,
    )

    assert f"-aime{effective_repeats}-" in name


def test_explicit_protocol_cannot_claim_a_different_repeat_count() -> None:
    with pytest.raises(ValueError, match="must contain aime8"):
        resolve_protocol_name(
            temperature="0.6",
            top_p="0.95",
            top_k=20,
            effective_repeats=8,
            requested_name=(
                "eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-"
                "t0.6-p0.95-k20-aime64-v1"
            ),
        )


def test_matching_custom_protocol_name_is_allowed() -> None:
    assert resolve_protocol_name(
        temperature="0.7",
        top_p="0.9",
        top_k=10,
        effective_repeats=8,
        requested_name="comparison-aime8-v2",
    ) == "comparison-aime8-v2"


def test_shell_entry_points_delegate_protocol_identity_to_the_helper() -> None:
    scripts = REPO_ROOT / "experiments" / "scripts" / "reasoning_eval"
    for name in ("run-evaluation.sbatch", "submit-staleness-sweep.sh", "show-results.sh"):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "protocol.py" in text
        assert "aime64-v1" not in text

    runner = (scripts / "run-evaluation.sbatch").read_text(encoding="utf-8")
    assert "#PBS -q R9920261300" in runner
    assert "#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1" in runner
    assert "#PBS -P" not in runner
    assert runner.index("evaluation-contract.env") < runner.index("declare -a PENDING_TASKS")
    assert "artifact-manifest.sha256" in runner
