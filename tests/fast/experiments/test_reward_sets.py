from __future__ import annotations

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.src.reward_sets import code, instruction_following, math_code_stem, stem, tau, tool_call
from miles.utils.types import Sample


def _run_reward(reward_function, sample_or_samples):
    return asyncio.run(reward_function(SimpleNamespace(), sample_or_samples))


def test_stem_reward_supports_scalar_and_mixed_batch_in_input_order(monkeypatch):
    monkeypatch.setenv("REASONING_GYM_ALLOW_EXACT_FALLBACK", "1")
    mcqa = Sample(
        response="Selected Option -> B",
        label="B",
        metadata={
            "verifier": "mcqa_regex",
            "output_regex": r"Selected Option\s*->\s*([A-D])",
            "valid_letters": ["A", "B", "C", "D"],
        },
    )
    reasoning = Sample(
        response="Answer: Richard",
        label="Richard",
        metadata={"verifier": "reasoning_gym"},
    )
    wrong = Sample(
        response="Selected Option -> A",
        label="B",
        metadata=mcqa.metadata,
    )

    assert _run_reward(stem.reward, mcqa) == 1.0
    assert _run_reward(stem.reward, [mcqa, reasoning, wrong]) == [1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "reward_module",
    [code, instruction_following, math_code_stem, stem, tau, tool_call],
)
def test_each_reward_set_preserves_scalar_and_batch_contract(monkeypatch, reward_module):
    verifier = next(iter(reward_module.ALLOWED_VERIFIERS))

    async def fake_handler(args, sample_or_samples, **kwargs):
        if isinstance(sample_or_samples, list):
            return [float(index) for index in range(len(sample_or_samples))]
        return 0.25

    monkeypatch.setitem(reward_module._HANDLERS, verifier, fake_handler)
    sample = Sample(response="", metadata={"verifier": verifier})

    assert _run_reward(reward_module.reward, sample) == 0.25
    assert _run_reward(reward_module.reward, [sample, sample]) == [0.0, 1.0]


@pytest.mark.parametrize(
    ("reward_function", "allowed_verifiers"),
    [
        (code.reward, {"python_code"}),
        (instruction_following.reward, {"ifeval_g"}),
        (math_code_stem.reward, {"math", "python_code", "mcqa_regex", "reasoning_gym"}),
        (stem.reward, {"mcqa_regex", "reasoning_gym", "gpqa"}),
        (tau.reward, {"expert_action"}),
        (tool_call.reward, {"expert_action"}),
    ],
)
def test_reward_sets_fail_closed_before_calling_a_handler(monkeypatch, reward_function, allowed_verifiers):
    module = sys.modules[reward_function.__module__]
    called = False

    async def unexpected_handler(args, sample_or_samples, **kwargs):
        nonlocal called
        called = True
        return 1.0

    first_allowed = next(iter(allowed_verifiers))
    monkeypatch.setitem(module._HANDLERS, first_allowed, unexpected_handler)
    samples = [
        Sample(response="", metadata={"verifier": first_allowed}),
        Sample(response="", metadata={"verifier": "calendar_constraints"}),
    ]

    with pytest.raises(ValueError, match="rejects verifier"):
        _run_reward(reward_function, samples)
    assert called is False


def test_tool_call_reward_accepts_only_exact_function_call():
    metadata = {
        "verifier": "expert_action",
        "expected_action": {"type": "function_call", "name": "search", "arguments": '{"q":"miles"}'},
    }
    correct = Sample(
        response='<tool_call>{"name":"search","arguments":{"q":"miles"}}</tool_call>',
        metadata=metadata,
    )
    wrong = Sample(
        response='<tool_call>{"name":"search","arguments":{"q":"other"}}</tool_call>',
        metadata=metadata,
    )

    assert _run_reward(tool_call.reward, [correct, wrong]) == [1.0, 0.0]


@pytest.mark.parametrize(
    ("module_name", "forbidden_modules"),
    [
        (
            "experiments.src.reward_sets.code",
            (
                "experiments.src.environments.instruction_following.verifier",
                "experiments.src.environments.reasoning_gym.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
        (
            "experiments.src.reward_sets.instruction_following",
            (
                "experiments.src.environments.competitive_programming.verifier",
                "experiments.src.environments.reasoning_gym.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
        (
            "experiments.src.reward_sets.stem",
            (
                "experiments.src.environments.competitive_programming.verifier",
                "experiments.src.environments.instruction_following.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
        (
            "experiments.src.reward_sets.math_code_stem",
            (
                "experiments.src.environments.instruction_following.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
        (
            "experiments.src.reward_sets.tool_call",
            (
                "experiments.src.environments.competitive_programming.verifier",
                "experiments.src.environments.instruction_following.verifier",
                "experiments.src.environments.reasoning_gym.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
        (
            "experiments.src.reward_sets.tau",
            (
                "experiments.src.environments.competitive_programming.verifier",
                "experiments.src.environments.instruction_following.verifier",
                "experiments.src.environments.reasoning_gym.verifier",
                "experiments.src.environments.calendar.verifier",
                "experiments.src.environments.workplace.runtime",
            ),
        ),
    ],
)
def test_reward_set_imports_are_domain_scoped(module_name, forbidden_modules):
    program = f"""
import sys
import {module_name}
for name in {forbidden_modules!r}:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", program], check=True)
