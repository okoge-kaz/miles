"""GPU smoke test for scheduler-authoritative SGLang policy provenance.

Run against a patched SGLang server started with ``--weight-version 10``.
The test opens a versioned begin/end session without transferring bytes, matching
the P2P lifecycle where weights are written out of band between those calls.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def _post(base_url: str, endpoint: str, payload: dict, timeout: float = 120.0):
    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _assert_success(result, endpoint: str) -> None:
    if isinstance(result, dict):
        success = result.get("success")
        message = result.get("message")
    else:
        success = result[0]
        message = result[1] if len(result) > 1 else ""
    assert success, f"{endpoint} failed: {message}"


def _generate(base_url: str, max_new_tokens: int) -> dict:
    result = _post(
        base_url,
        "generate",
        {
            "text": "Count upward slowly: 1, 2, 3,",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=300.0,
    )
    return result["meta_info"]


def _assert_versions(meta: dict, *, first: int, minimum: int, maximum: int, last: int) -> None:
    expected = {
        "first_prefill_weight_version": first,
        "min_forward_weight_version": minimum,
        "max_forward_weight_version": maximum,
        "last_forward_weight_version": last,
    }
    actual = {key: meta.get(key) for key in expected}
    assert actual == expected, f"policy provenance mismatch: expected={expected}, actual={actual}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--pause-delay", type=float, default=0.5)
    args = parser.parse_args()

    baseline = _generate(args.base_url, max_new_tokens=8)
    _assert_versions(baseline, first=10, minimum=10, maximum=10, last=10)

    _post(args.base_url, "slow_down", {"forward_sleep_time": 0.01})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate, args.base_url, 512)
        time.sleep(args.pause_delay)
        _post(args.base_url, "pause_generation", {"mode": "in_place"})

        begin = _post(
            args.base_url,
            "begin_weight_update",
            {"selector": "all", "weight_version": "11"},
        )
        _assert_success(begin, "begin_weight_update")
        end = _post(args.base_url, "end_weight_update", {})
        _assert_success(end, "end_weight_update")
        _post(args.base_url, "continue_generation", {})
        mixed = future.result(timeout=300.0)

    _assert_versions(mixed, first=10, minimum=10, maximum=11, last=11)
    assert mixed["response_weight_version"] == "11"
    assert mixed["weight_version"] == "11"

    _post(args.base_url, "slow_down", {"forward_sleep_time": None})
    refreshed = _generate(args.base_url, max_new_tokens=8)
    _assert_versions(refreshed, first=11, minimum=11, maximum=11, last=11)

    print(
        json.dumps(
            {
                "baseline": baseline,
                "mixed": mixed,
                "refreshed": refreshed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
