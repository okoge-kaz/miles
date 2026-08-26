"""Shared user-simulator primitives for Tau Bench training and evaluation."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_GENERATE_CONTENT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
STOP_MARKER = "###STOP###"

_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class UserGeneration:
    """One simulated user turn plus API accounting metadata."""

    text: str
    output_tokens: int
    finish_reason: str


class GeminiRequestError(RuntimeError):
    """A sanitized Gemini user-simulator request failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


GeminiPost = Callable[
    [str, dict[str, Any], dict[str, str]],
    Awaitable[dict[str, Any]],
]


def build_user_system_prompt(instruction: str) -> str:
    """Build the user simulator prompt from the official Tau interaction rules."""

    return f"""You simulate the user in a customer-service conversation.

Private task instruction:
{instruction}

Rules:
- Generate exactly one natural user message at a time.
- Reveal only information needed for the current step.
- Never invent information absent from the private instruction.
- Do not quote the private instruction verbatim.
- Emit {STOP_MARKER} alone only after the goal is satisfied and execution is confirmed.
- Do not act as the customer-service agent and do not call tools."""


def require_gemini_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Read the Gemini key from the process environment and fail closed if absent."""

    source = os.environ if environ is None else environ
    api_key = source.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiRequestError(
            "GEMINI_API_KEY is required for the Gemini Tau user backend; "
            "export it in the process/job environment (.env files are not read)"
        )
    return api_key


def gemini_generate_content_url(model: str) -> str:
    """Return the fixed-host Gemini REST endpoint for a validated model name."""

    if not _MODEL_NAME_PATTERN.fullmatch(model):
        raise ValueError(f"invalid Gemini model name {model!r}")
    return f"{GEMINI_GENERATE_CONTENT_BASE}/{model}:generateContent"


def build_gemini_payload(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    """Convert OpenAI-style user-simulator history to Gemini REST JSON."""

    if max_output_tokens < 1:
        raise ValueError("Gemini max_output_tokens must be positive")
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    role_map = {"user": "user", "assistant": "model"}
    for message in messages:
        role = str(message.get("role") or "")
        text = str(message.get("content") or "")
        if role == "system":
            system_parts.append({"text": text})
            continue
        if role not in role_map:
            raise ValueError(f"unsupported Gemini user-simulator message role {role!r}")
        contents.append({"role": role_map[role], "parts": [{"text": text}]})
    if not system_parts:
        raise ValueError("Gemini user-simulator history requires a system instruction")
    if not contents:
        raise ValueError("Gemini user-simulator history requires conversation content")
    return {
        "systemInstruction": {"parts": system_parts},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "topP": top_p,
            "seed": seed,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def parse_gemini_response(response: Mapping[str, Any]) -> UserGeneration:
    """Parse one text candidate without copying prompt or credential data into errors."""

    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = response.get("promptFeedback")
        block_reason = prompt_feedback.get("blockReason") if isinstance(prompt_feedback, Mapping) else None
        detail = f" (blockReason={block_reason})" if block_reason else ""
        raise GeminiRequestError(f"Gemini returned no user-simulator candidate{detail}")

    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise GeminiRequestError("Gemini returned a malformed user-simulator candidate")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        parts = []
    text = "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping) and not bool(part.get("thought"))
    ).strip()
    finish_reason = str(candidate.get("finishReason") or "UNKNOWN")
    if not text:
        raise GeminiRequestError(f"Gemini returned an empty user message (finishReason={finish_reason})")

    usage = response.get("usageMetadata")
    output_tokens = int(usage.get("candidatesTokenCount") or 0) if isinstance(usage, Mapping) else 0
    normalized_text = STOP_MARKER if STOP_MARKER in text else text
    return UserGeneration(text=normalized_text, output_tokens=output_tokens, finish_reason=finish_reason)


def _status_code(error: BaseException) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _is_retryable(error: BaseException) -> bool:
    status_code = _status_code(error)
    if isinstance(error, GeminiRequestError) and status_code is None:
        return False
    return status_code is None or status_code in {408, 409, 429} or status_code >= 500


async def generate_gemini_user(
    messages: Sequence[Mapping[str, Any]],
    *,
    post_json: GeminiPost,
    model: str = DEFAULT_GEMINI_MODEL,
    max_output_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 0,
    request_timeout: float = 120.0,
    max_retries: int = 4,
    retry_backoff: float = 1.0,
) -> UserGeneration:
    """Generate one Tau user turn through Gemini's fixed-host REST API."""

    if request_timeout <= 0:
        raise ValueError("Gemini request_timeout must be positive")
    if max_retries < 0:
        raise ValueError("Gemini max_retries must be non-negative")
    if retry_backoff < 0:
        raise ValueError("Gemini retry_backoff must be non-negative")

    api_key = require_gemini_api_key()
    url = gemini_generate_content_url(model)
    payload = build_gemini_payload(
        messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(post_json(url, payload, headers), timeout=request_timeout)
            return parse_gemini_response(response)
        except Exception as error:
            can_retry = attempt < max_retries and _is_retryable(error)
            if not can_retry:
                status_code = _status_code(error)
                detail = f"HTTP {status_code}" if status_code is not None else type(error).__name__
                raise GeminiRequestError(
                    f"Gemini user generation failed after {attempt + 1} attempt(s): {detail}",
                    status_code=status_code,
                ) from error
            delay = retry_backoff * (2**attempt)
            logger.warning(
                "Gemini user generation attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                f"HTTP {_status_code(error)}" if _status_code(error) is not None else type(error).__name__,
                delay,
            )
            if delay:
                await asyncio.sleep(delay)

    raise AssertionError("Gemini retry loop exited unexpectedly")
