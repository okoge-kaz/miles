from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.src.environments.competitive_programming import verifier as code_exec
from experiments.src.reward_sets import code
from miles.utils.types import Sample


def _reward(sample_or_samples):
    return asyncio.run(code.reward(SimpleNamespace(), sample_or_samples))


def test_code_verifier_discriminates_stdin_programs(monkeypatch):
    monkeypatch.setattr(code_exec, "SANDBOX_BACKEND", "process")
    tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
    correct = "a, b = map(int, input().split())\nprint(a + b)"
    wrong = "print(0)"

    assert code_exec.run_tests(correct, tests, timeout=2) == 1.0
    assert code_exec.run_tests(wrong, tests, timeout=2) == 0.0


def test_code_verifier_supports_published_harness(monkeypatch):
    monkeypatch.setattr(code_exec, "SANDBOX_BACKEND", "process")
    tests = {
        "entry_point": "Solution().add",
        "import_prefix": "from typing import *\n",
        "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }
    correct = "class Solution:\n    def add(self, left, right):\n        return left + right"
    wrong = "class Solution:\n    def add(self, left, right):\n        return 0"

    assert code_exec.run_tests(correct, tests, timeout=2) == 1.0
    assert code_exec.run_tests(wrong, tests, timeout=2) == 0.0


def test_scalar_code_rewards_share_execution_limit(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_score(sample):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 1.0

    monkeypatch.setattr(code_exec, "CONCURRENCY", 2)
    monkeypatch.setattr(code_exec, "_score_one", fake_score)
    code_exec._LOOP_SEMAPHORES.clear()
    sample = Sample(response="pass", metadata={"verifier": "python_code"})

    async def run_scalar_contract():
        return await asyncio.gather(*(code_exec.code_exec_reward(None, sample) for _ in range(8)))

    assert asyncio.run(run_scalar_contract()) == [1.0] * 8
    assert max_active == 2


def test_code_reward_preserves_scalar_and_batch_contract(monkeypatch):
    async def fake_handler(args, sample_or_samples, **kwargs):
        if isinstance(sample_or_samples, list):
            return [float(index) for index in range(len(sample_or_samples))]
        return 0.25

    monkeypatch.setitem(code._HANDLERS, "python_code", fake_handler)
    sample = Sample(response="", metadata={"verifier": "python_code"})

    assert _reward(sample) == 0.25
    assert _reward([sample, sample]) == [0.0, 1.0]


def test_code_reward_fails_closed_before_dispatch(monkeypatch):
    called = False

    async def unexpected_handler(args, sample_or_samples, **kwargs):
        nonlocal called
        called = True
        return 1.0

    monkeypatch.setitem(code._HANDLERS, "python_code", unexpected_handler)
    samples = [
        Sample(response="", metadata={"verifier": "python_code"}),
        Sample(response="", metadata={"verifier": "reasoning_gym"}),
    ]

    with pytest.raises(ValueError, match="rejects verifier"):
        _reward(samples)
    assert called is False
