import asyncio
import json
import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)
EVENTS = Counter()


@lru_cache(maxsize=1)
def endpoints() -> dict:
    path = Path(os.environ["NEMO_GYM_ENDPOINTS"])
    value = json.loads(path.read_text())
    if value.get("schema") != 1:
        raise ValueError("NeMo Gym endpoint manifest must use schema 1")
    return value


class ScoringFailure(ValueError):
    def __init__(self, kind, *, retryable=True, http_status=None):
        super().__init__(kind)
        self.kind = kind if isinstance(kind, str) and re.fullmatch(r"[a-z0-9_]{1,80}", kind) else "upstream_failure"
        self.retryable = retryable
        self.http_status = http_status


def validate_result(result, *, reward_key="reward"):
    if not isinstance(result, dict):
        raise ScoringFailure("invalid_response")
    if result.get("error_type") == "schema_error":
        raise ScoringFailure("invalid_schema", retryable=False)
    if result.get("error_type") == "api_error":
        raise ScoringFailure("model_api_error")
    if result.get("mask_sample") or result.get("error") or result.get("_ng_failure_class"):
        kind = result.get("failure_kind") or result.get("_ng_failure_class") or "upstream_failure"
        if kind == "judge_failed" and result.get("failure_reason") == "judge_invalid_response":
            kind = "judge_invalid_response"
        raise ScoringFailure(
            kind,
            retryable=result.get("retryable", True),
        )
    score = result.get(reward_key)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ScoringFailure("invalid_reward")
    if reward_key == "answer_em" and score not in (0, 1):
        raise ScoringFailure("invalid_reward")


async def request(
    service: str, route: str, body: dict, *, attempts: int = 3, diagnostics=None, reward_key="reward"
) -> dict | None:
    """Retry stateless verification; stateful callers use one attempt per fresh episode."""
    url = endpoints()["services"][service].rstrip("/") + route
    timeout = float(os.environ.get("NEMO_GYM_REQUEST_TIMEOUT", "600"))
    for attempt in range(1, attempts + 1):
        EVENTS[f"{service}/request_count"] += 1
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False) as session:
                async with session.post(url, json=body, allow_redirects=False) as response:
                    if response.status != 200:
                        try:
                            error_body = await response.json()
                        except (ValueError, aiohttp.ContentTypeError):
                            error_body = {}
                        detail = error_body.get("detail", {}) if isinstance(error_body, dict) else {}
                        if isinstance(detail, dict) and detail.get("failure_kind"):
                            raise ScoringFailure(
                                detail["failure_kind"],
                                retryable=detail.get("retryable", True),
                                http_status=response.status,
                            )
                        response.raise_for_status()
                        raise ScoringFailure("http_error", http_status=response.status)
                    result = await response.json()
            validate_result(result, reward_key=reward_key)
            EVENTS[f"{service}/scored_count"] += 1
            if diagnostics is not None:
                diagnostics.update(
                    status="scored", service=service, attempts=attempt, reward=float(result[reward_key])
                )
                diagnostics.pop("failure_kind", None)
                diagnostics.pop("http_status", None)
            return result
        except (aiohttp.ClientError, TimeoutError, ValueError) as error:
            kind = getattr(error, "kind", None) or (
                "http_timeout"
                if isinstance(error, TimeoutError)
                else (
                    "http_error"
                    if isinstance(error, aiohttp.ClientResponseError)
                    else "transport_error"
                    if isinstance(error, aiohttp.ClientError)
                    else "invalid_response"
                )
            )
            http_status = getattr(error, "http_status", getattr(error, "status", None))
            EVENTS[f"{service}/failure/{kind}"] += 1
            if diagnostics is not None:
                diagnostics.update(
                    status="unscored",
                    service=service,
                    attempts=attempt,
                    reward=None,
                    failure_kind=kind,
                    http_status=http_status,
                )
            # Avoid logging raw HTTP bodies containing prompts, credentials or task state.
            logger.warning(
                "NeMo Gym scoring failed: service=%s attempt=%d/%d failure_kind=%s http_status=%s",
                service,
                attempt,
                attempts,
                kind,
                http_status,
            )
            if not getattr(error, "retryable", True):
                break
            if attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
    EVENTS[f"{service}/unscored_count"] += 1
    logger.warning("NeMo Gym sample excluded: service=%s attempts=%d failure_kind=%s", service, attempt, kind)
    return None
