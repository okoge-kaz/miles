# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export NeMo Evaluator adapter request/response caches as readable JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from nemo_evaluator.adapters.caching.diskcaching import Cache


LOGGER = logging.getLogger(__name__)


def _decode_json(value: Any) -> Any:
    """Decode a cached JSON payload while preserving non-JSON values."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw_text": value}
    return value


def export_cache(*, cache_dir: Path, output: Path) -> int:
    """Join cached requests and responses by cache key and write JSONL."""
    request_dir = cache_dir / "requests"
    response_dir = cache_dir / "responses"
    response_database = response_dir / "cache.db"
    if not response_database.is_file():
        raise FileNotFoundError(f"response cache not found: {response_database}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.partial-{os.getpid()}")
    request_cache = Cache(directory=str(request_dir))
    response_cache = Cache(directory=str(response_dir))
    count = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as stream:
            for cache_key in response_cache:
                try:
                    request = _decode_json(request_cache[cache_key])
                except KeyError:
                    request = None
                response = _decode_json(response_cache[cache_key])
                record = {
                    "cache_key": (
                        cache_key.decode("utf-8", errors="replace")
                        if isinstance(cache_key, bytes)
                        else cache_key
                    ),
                    "request": request,
                    "response": response,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        request_cache.close()
        response_cache.close()

    if count == 0:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"response cache is empty: {response_dir}")
    os.replace(temporary_output, output)
    return count


def main() -> None:
    """Parse command-line arguments and export the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_cache(cache_dir=args.cache_dir, output=args.output)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Exported %d model request/response records to %s", count, args.output)


if __name__ == "__main__":
    main()
