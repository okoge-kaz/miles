"""Verify that a Ray Jobs driver and worker inherit a raylet environment value."""

from __future__ import annotations

import os
import secrets

import ray

SENTINEL_ENV_NAME = "MILES_TAU_RAY_SECRET_SENTINEL"


@ray.remote
def _worker_observes(expected: str) -> bool:
    observed = os.environ.get(SENTINEL_ENV_NAME, "")
    return bool(observed) and secrets.compare_digest(observed, expected)


def main() -> None:
    expected = os.environ.get(SENTINEL_ENV_NAME, "")
    if not expected:
        raise RuntimeError(f"Ray Jobs driver did not inherit {SENTINEL_ENV_NAME}")
    ray.init(address="auto")
    if not ray.get(_worker_observes.remote(expected)):
        raise RuntimeError(f"Ray worker did not inherit {SENTINEL_ENV_NAME} from its raylet")
    print("Ray cluster environment inheritance: driver=ok worker=ok")


if __name__ == "__main__":
    main()
