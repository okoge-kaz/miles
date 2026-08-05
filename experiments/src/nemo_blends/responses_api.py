"""Translate OpenAI Responses-API conversations into the chat form Qwen templates read.

The NeMo Gym tool-use components store conversations as Responses-API *items*
(`message`, `reasoning`, `function_call`, `function_call_output`) rather than
ChatCompletions messages, and their tools use the flat Responses shape. Qwen's
chat template understands neither, so both need translating before
`apply_chat_template`.

What the Qwen3 template actually consumes (verified against
Qwen3-4B-Instruct-2507's tokenizer_config):

  * `tools`  -- dumped verbatim with `tool | tojson` inside <tools></tools>.
    Any shape renders, but the nested ChatCompletions form is what the model was
    trained on, so that is what we emit.
  * assistant tool calls -- `message.tool_calls[].function.{name,arguments}`,
    rendered as `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
    A string `arguments` is inserted raw, so the JSON text from the dataset can
    pass straight through.
  * tool results -- `role: "tool"`, rendered inside <tool_response></tool_response>.

`reasoning` items are dropped. They hold the *expert* model's private chain of
thought, and the policy being trained has no channel to put them in: Qwen3
Instruct-2507 emits no thinking block, and the template has nowhere to render
one. Keeping them would show the policy reasoning it is not asked to reproduce.
"""

import json

__all__ = ["to_chat_messages", "to_chat_tools", "expected_action_signature"]


def to_chat_tools(tools):
    """Responses-API tool specs -> ChatCompletions `{"type":"function","function":{...}}`."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        # Already nested: pass through untouched.
        if isinstance(tool.get("function"), dict):
            out.append({"type": "function", "function": tool["function"]})
            continue
        name = tool.get("name")
        if not name:
            continue
        fn = {"name": name}
        if tool.get("description"):
            fn["description"] = tool["description"]
        # `strict` is a Responses-API execution flag, not part of the signature.
        fn["parameters"] = tool.get("parameters") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": fn})
    return out


def _text_of(content):
    """Content is a string, or a list of Responses-API content parts."""
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


def to_chat_messages(items):
    """Responses-API input items -> ChatCompletions messages."""
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
                "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or "{}"},
            }
            # Consecutive calls belong to one assistant turn; the template
            # renders several <tool_call> blocks inside a single message.
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

        # Plain message item, or a bare {role, content} dict.
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": _text_of(item.get("content"))})

    return messages


def expected_action_signature(action):
    """The comparable form of an expected_action.

    Reduces the expert's action to what a policy could reproduce: for a tool call
    the function name plus its arguments as a parsed dict, for a plain reply just
    the marker that no call was expected. Ids, statuses and call_ids are dropped
    -- they are per-trajectory bookkeeping the policy cannot and should not match.
    """
    if not isinstance(action, dict):
        return None
    if action.get("type") == "function_call":
        raw = action.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            arguments = {"__unparsed__": raw}
        return {"kind": "function_call", "name": action.get("name") or "", "arguments": arguments}
    if action.get("type") == "message":
        return {"kind": "message", "content": _text_of(action.get("content"))}
    return None
