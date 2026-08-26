from __future__ import annotations

import json
import subprocess
import sys

from experiments.src.datasets.calendar.prepare import adapt_calendar
from experiments.src.environments.calendar.verifier import (
    build_calendar_solution,
    score_calendar_response,
)


def _state() -> dict[str, dict[str, object]]:
    return {
        "0": {
            "event_id": 0,
            "duration": 60,
            "constraint": "between 10am and 11:30am",
            "min_time": "10:00",
            "max_time": "16:00",
        },
        "1": {
            "event_id": 1,
            "duration": 30,
            "constraint": "after 11am",
            "min_time": "10:00",
            "max_time": "16:00",
        },
    }


def test_calendar_conversion_is_deterministic_and_locally_verifiable():
    state = _state()
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Return the calendar as JSON."},
                {"role": "user", "content": "Schedule both events."},
            ]
        },
        "exp_cal_state": state,
    }
    first = adapt_calendar(row, eval_only=False)
    second = adapt_calendar(row, eval_only=False)

    assert first == second
    assert first["metadata"]["verifier"] == "calendar_constraints"
    assert first["metadata"]["eval_only"] is False
    assert first["metadata"]["runtime_dependency"] == "miles-local-calendar-verifier"
    assert json.loads(first["label"]) == state
    assert score_calendar_response(build_calendar_solution(state), state) == 1.0


def test_calendar_verifier_rejects_overlap_and_constraint_violations():
    state = _state()
    overlapping = json.dumps(
        [
            {"event_id": 0, "event_name": "a", "start_time": "10:00", "duration": 60},
            {"event_id": 1, "event_name": "b", "start_time": "10:30", "duration": 30},
        ]
    )
    assert score_calendar_response(overlapping, state) == 0.0


def test_calendar_dataset_import_does_not_load_rl_runtime():
    program = """
import sys
import experiments.src.datasets.calendar.prepare
assert not any(name == 'miles' or name.startswith('miles.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", program], check=True)
