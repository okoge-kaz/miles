"""Verify Nemotron Calendar schedules locally from their expected state."""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections.abc import Mapping
from typing import Any

_CLOCK_PATTERN = re.compile(r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<period>am|pm)?$", re.IGNORECASE)


def _time_to_minutes(value: Any) -> int:
    text = str(value or "").strip().lower().replace(" ", "")
    match = _CLOCK_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid clock time: {value!r}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    period = match.group("period")
    if minute >= 60:
        raise ValueError(f"invalid clock time: {value!r}")
    if period:
        if not 1 <= hour <= 12:
            raise ValueError(f"invalid 12-hour time: {value!r}")
        hour = hour % 12 + (12 if period.lower() == "pm" else 0)
    elif hour >= 24:
        raise ValueError(f"invalid 24-hour time: {value!r}")
    return hour * 60 + minute


def _extract_json_list(response: str) -> list[Any] | None:
    decoder = json.JSONDecoder()
    text = str(response or "")
    for start, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def _constraint_satisfied(start: int, end: int, constraint: Any) -> bool:
    if constraint is None:
        return True
    text = str(constraint).strip().lower()
    if text.startswith("before "):
        return end <= _time_to_minutes(text.removeprefix("before "))
    if text.startswith("after "):
        return start >= _time_to_minutes(text.removeprefix("after "))
    if text.startswith("at "):
        return start == _time_to_minutes(text.removeprefix("at "))
    if text.startswith("between "):
        bounds = text.removeprefix("between ").split(" and ")
        if len(bounds) != 2:
            return False
        return start >= _time_to_minutes(bounds[0]) and end <= _time_to_minutes(bounds[1])
    return False


def _event_interval(event: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[int, int]:
    duration = event.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError("event duration must be an integer")
    if duration != expected.get("duration"):
        raise ValueError("event duration differs from the expected duration")
    start = _time_to_minutes(event.get("start_time"))
    end = start + duration
    minimum = _time_to_minutes(expected.get("min_time"))
    maximum = _time_to_minutes(expected.get("max_time"))
    if start < minimum or end > maximum:
        raise ValueError("event is outside its scheduling window")
    if not _constraint_satisfied(start, end, expected.get("constraint")):
        raise ValueError("event violates its scheduling constraint")
    return start, end


def _normalize_events(events: list[Any], expected_ids: set[str]) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("calendar entries must be objects")
        event_id = str(event.get("event_id"))
        if event_id in normalized:
            raise ValueError(f"duplicate event id: {event_id}")
        normalized[event_id] = event
    if set(normalized) != expected_ids:
        raise ValueError("calendar event ids differ from the expected state")
    return normalized


def score_calendar_response(response: str, expected_state: Any) -> float:
    """Return one only for a complete, non-overlapping constraint-satisfying schedule."""

    if "<think>" in str(response or ""):
        return 0.0
    if not isinstance(expected_state, Mapping) or not expected_state:
        return 0.0
    events = _extract_json_list(response)
    if events is None:
        return 0.0
    try:
        expected = {str(key): value for key, value in expected_state.items()}
        normalized = _normalize_events(events, set(expected))
        intervals = []
        for event_id, event in normalized.items():
            if not isinstance(expected[event_id], Mapping):
                raise ValueError("expected calendar entry must be an object")
            start, end = _event_interval(event, expected[event_id])
            intervals.append((start, end))
        intervals.sort()
        if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
            return 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0
    return 1.0


def _allowed_starts(expected: Mapping[str, Any]) -> list[int]:
    duration = int(expected["duration"])
    minimum = _time_to_minutes(expected["min_time"])
    maximum = _time_to_minutes(expected["max_time"])
    return [
        start
        for start in range(minimum, maximum - duration + 1, 5)
        if _constraint_satisfied(start, start + duration, expected.get("constraint"))
    ]


def build_calendar_solution(expected_state: Mapping[str, Any]) -> str:
    """Build one deterministic valid schedule for reward preflight and data audit."""

    expected = {str(key): value for key, value in expected_state.items()}
    domains = {event_id: _allowed_starts(event) for event_id, event in expected.items()}
    assigned: dict[str, int] = {}

    def assign(remaining: tuple[str, ...], cursor: int) -> bool:
        if not remaining:
            return True
        ordered = sorted(
            remaining,
            key=lambda event_id: (domains[event_id][-1] if domains[event_id] else -1, event_id),
        )
        for event_id in ordered:
            starts = domains[event_id]
            index = bisect_left(starts, cursor)
            if index == len(starts):
                continue
            start = starts[index]
            duration = int(expected[event_id]["duration"])
            end = start + duration
            assigned[event_id] = start
            next_remaining = tuple(candidate for candidate in remaining if candidate != event_id)
            if assign(next_remaining, end):
                return True
            del assigned[event_id]
        return False

    if not assign(tuple(expected), min(_time_to_minutes(event["min_time"]) for event in expected.values())):
        raise ValueError("expected calendar state has no feasible schedule")
    events = [
        {
            "event_id": int(event_id) if event_id.isdigit() else event_id,
            "event_name": f"event-{event_id}",
            "start_time": f"{assigned[event_id] // 60:02d}:{assigned[event_id] % 60:02d}",
            "duration": int(expected[event_id]["duration"]),
        }
        for event_id in sorted(expected, key=lambda value: (assigned[value], value))
    ]
    return json.dumps(events, separators=(",", ":"))
