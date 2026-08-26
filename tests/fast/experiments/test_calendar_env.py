from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from experiments.src.datasets.calendar.prepare import adapt_calendar
from experiments.src.environments.calendar.verifier import build_calendar_solution, score_calendar_response
from experiments.src.reward_sets.all_domains import blend_reward
from miles.utils.types import Sample


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


def test_calendar_solution_and_reward_reject_overlap_and_constraint_violation():
    state = _state()
    solution = build_calendar_solution(state)
    assert score_calendar_response(solution, state) == 1.0
    overlapping = json.dumps(
        [
            {"event_id": 0, "event_name": "a", "start_time": "10:00", "duration": 60},
            {"event_id": 1, "event_name": "b", "start_time": "10:30", "duration": 30},
        ]
    )
    assert score_calendar_response(overlapping, state) == 0.0


def test_calendar_adapter_is_local_and_blend_reward_dispatches():
    state = _state()
    converted = adapt_calendar(
        {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Return the calendar as JSON."},
                    {"role": "user", "content": "Schedule both events."},
                ]
            },
            "exp_cal_state": state,
        },
        eval_only=False,
    )
    assert converted["metadata"]["runtime_dependency"] == "miles-local-calendar-verifier"
    assert "nemo" not in converted["metadata"]["runtime_dependency"]
    sample = Sample(
        response=build_calendar_solution(state),
        label=converted["label"],
        metadata=converted["metadata"],
    )
    assert asyncio.run(blend_reward(SimpleNamespace(), sample)) == 1.0
