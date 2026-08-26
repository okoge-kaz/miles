"""Translate OpenAI Responses API records into chat-template inputs."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["expected_action_signature", "to_chat_messages", "to_chat_tools"]


def to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert flat Responses API tools to Chat Completions tool specs."""
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            converted.append({"type": "function", "function": tool["function"]})
            continue
        if not tool.get("name"):
            continue
        function = {
            "name": tool["name"],
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


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


def expected_action_signature(action: Any) -> dict[str, Any] | None:
    """Reduce an expert action to fields a policy can reproduce."""
    if not isinstance(action, dict):
        return None
    action_type = action.get("type")
    if action_type == "function_call" or (action.get("name") and "arguments" in action):
        raw = action.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            arguments = {"__unparsed__": raw}
        return {
            "kind": "function_call",
            "name": action.get("name") or "",
            "arguments": arguments,
        }
    if action_type == "message" or "content" in action:
        return {"kind": "message", "content": _text_of(action.get("content"))}
    return None
