#!/usr/bin/env python3
"""Resolve a reasoning-evaluation protocol name from its sampling controls."""

from __future__ import annotations

import argparse
import re


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_DECIMAL = re.compile(r"^[0-9]+(?:[.][0-9]+)?$")
_PROTOCOL_PREFIX = "eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl"


def derive_protocol_name(
    *,
    temperature: str,
    top_p: str,
    top_k: int,
    effective_repeats: int,
    enable_thinking: bool = False,
) -> str:
    """Return the canonical protocol name for the effective sampling controls."""
    if not _DECIMAL.fullmatch(temperature) or not _DECIMAL.fullmatch(top_p):
        raise ValueError("temperature and top-p must be nonnegative decimals")
    if top_k <= 0 or effective_repeats <= 0:
        raise ValueError("top-k and effective repeats must be positive")
    reasoning_mode = "thinking" if enable_thinking else "instruct"
    return (
        f"{_PROTOCOL_PREFIX}-{reasoning_mode}-t{temperature}-p{top_p}-k{top_k}"
        f"-aime{effective_repeats}-v1"
    )


def resolve_protocol_name(
    *,
    temperature: str,
    top_p: str,
    top_k: int,
    effective_repeats: int,
    enable_thinking: bool = False,
    requested_name: str | None = None,
) -> str:
    """Derive a name or validate that an explicit name records the repeat count."""
    derived_name = derive_protocol_name(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        effective_repeats=effective_repeats,
        enable_thinking=enable_thinking,
    )
    if not requested_name:
        return derived_name
    if not _SAFE_NAME.fullmatch(requested_name):
        raise ValueError("protocol name contains unsupported characters")
    repeat_tag = re.compile(rf"(?:^|-)aime{effective_repeats}(?:-|$)")
    if not repeat_tag.search(requested_name):
        raise ValueError(
            f"protocol name must contain aime{effective_repeats} for the effective repeat count"
        )
    return requested_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--top-p", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--effective-repeats", type=int, required=True)
    reasoning_mode = parser.add_mutually_exclusive_group()
    reasoning_mode.add_argument("--enable-thinking", action="store_true")
    reasoning_mode.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--protocol-name")
    return parser.parse_args()


def main() -> None:
    """Resolve a protocol name for shell callers."""
    args = _parse_args()
    try:
        protocol_name = resolve_protocol_name(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            effective_repeats=args.effective_repeats,
            enable_thinking=args.enable_thinking and not args.disable_thinking,
            requested_name=args.protocol_name,
        )
    except ValueError as error:
        raise SystemExit(f"invalid reasoning-evaluation protocol: {error}") from error
    print(protocol_name)


if __name__ == "__main__":
    main()
