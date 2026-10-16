from __future__ import annotations

from typing import Any

__all__ = ["to_chat_messages"]


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(content)


def to_chat_messages(items: Any) -> list[dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages."""
    messages = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "reasoning":
            continue
        if item_type == "function_call":
            call = {
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "{}",
                },
            }
            if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("content"):
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": _text_of(item.get("output")),
                }
            )
            continue
        if role in {"system", "user", "assistant", "tool"}:
            message = {"role": role, "content": _text_of(item.get("content"))}
            if role == "tool" and item.get("tool_call_id"):
                message["tool_call_id"] = item["tool_call_id"]
            messages.append(message)
    return messages
