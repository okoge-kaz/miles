"""Verifiers for the Nemotron RL blend components miles has no built-in rm_type for.

Each is mounted with `--custom-rm-path`, e.g.

    --custom-rm-path experiments.src.nemo_blends.rewards.reasoning_gym_reward

miles has *two* custom-rm contracts and reaches them by different routes:
`batched_async_rm` calls `fn(args, samples)` with a list, while `async_rm` calls
`fn(args, sample)` with one. The training path goes through the batched one, so
each entry point here accepts either and dispatches -- a function written for
only one contract silently receives a list where it expects a Sample.

None of these needs a container, an external service, or a second model. That is
the whole reason these particular components were picked: the ground truth
either *is* the answer (ReasoningGym), or is a machine-checkable artifact that
travels with the row (a JSON schema, an expert tool call, a set of unit tests).
"""

import json
import re
import string

from experiments.src.nemo_blends.responses_api import expected_action_signature

__all__ = [
    "reasoning_gym_reward",
    "structured_output_reward",
    "tool_call_match_reward",
]


# --------------------------------------------------------------------------
# ReasoningGym
# --------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation.replace("-", ""))


def _normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    # Strip a trailing "answer:" preamble and surrounding markup before comparing.
    text = re.sub(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", "", text)
    text = text.strip().strip("`*_ \t\n")
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    text = text.translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def _last_nonempty_line(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        if line.strip():
            return line
    return ""


async def _reasoning_gym_one(args, sample) -> float:
    """nvidia/Nemotron-RL-ReasoningGym-v1: the label is the literal answer.

    Rows carry a plain answer string ("Richard", "42", ...) rather than a boxed
    expression, so this compares the tail of the response against it after light
    normalisation. Checking the last line first and only then falling back to a
    whole-response containment test keeps a model that restates the question from
    being credited for echoing the answer inside its reasoning.
    """
    label = _normalize_answer(sample.label or "")
    if not label:
        return 0.0
    response = sample.response or ""

    if _normalize_answer(_last_nonempty_line(response)) == label:
        return 1.0

    boxed = re.findall(r"\\boxed\{([^}]*)\}", response)
    if boxed and _normalize_answer(boxed[-1]) == label:
        return 1.0

    tail = _normalize_answer(response[-400:])
    marker = re.search(r"(?:final\s+)?answer\s*[:\-]\s*(.+)", response[-400:], flags=re.IGNORECASE)
    if marker and _normalize_answer(marker.group(1).splitlines()[0]) == label:
        return 1.0
    # Last resort: a short answer appearing verbatim in the tail. Bounded to
    # short labels so a one-word answer is not matched by accident in prose.
    if len(label) >= 2 and label in tail and len(label.split()) <= 3:
        return 1.0
    return 0.0


# --------------------------------------------------------------------------
# structured outputs
# --------------------------------------------------------------------------


def _extract_json_object(text: str):
    """The JSON the model meant to emit, from a possibly chatty response."""
    text = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace-balanced span.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def _structured_output_one(args, sample) -> float:
    """nvidia/...instruction_following-structured_outputs: does the output validate?

    The row ships the JSON Schema it demands, so correctness is decidable without
    a reference answer: parse the response, validate against the schema. Only
    schema conformance is scored -- the task is formatting under constraints, and
    the content has no single right answer.
    """
    schema_str = (sample.metadata or {}).get("schema_str") or sample.label
    if not schema_str:
        return 0.0
    try:
        schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
    except json.JSONDecodeError:
        return 0.0

    obj = _extract_json_object(sample.response)
    if obj is None:
        return 0.0

    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment problem, not data
        raise RuntimeError("structured_output_reward needs `pip install jsonschema`") from exc

    # Some rows use `$ref: "#/$defs/X"` while defining X under the Draft-7
    # `definitions` key. jsonschema then raises a *referencing* error that is
    # neither ValidationError nor SchemaError, so a narrow except lets it escape
    # and kill the whole sweep. Normalising the alias fixes the common case;
    # the broad except covers whatever else a generated schema does wrong.
    if isinstance(schema, dict) and "definitions" in schema and "$defs" not in schema:
        schema = {**schema, "$defs": schema["definitions"]}

    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError:
        return 0.0
    except Exception:  # noqa: BLE001 - a broken schema is a bad row, not a model failure
        return 0.0
    return 1.0


# --------------------------------------------------------------------------
# tool call matching
# --------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_emitted_tool_calls(response: str):
    """Tool calls Qwen emitted, in the `<tool_call>{...}</tool_call>` form its template documents."""
    calls = []
    for blob in _TOOL_CALL_RE.findall(response or ""):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__unparsed__": arguments}
        calls.append({"name": name, "arguments": arguments or {}})
    return calls


def _arguments_match(expected: dict, actual: dict) -> bool:
    """Exact match on the expert's argument set, compared as JSON values.

    Not a subset test: an extra argument changes what the tool does, and a
    missing one is a different call. Values are compared after a str() round so
    that 42 and "42" -- which the two sides encode inconsistently -- agree.
    """
    if set(expected) != set(actual):
        return False
    for key, want in expected.items():
        got = actual[key]
        if want == got:
            continue
        if isinstance(want, (str, int, float, bool)) and isinstance(got, (str, int, float, bool)):
            if str(want).strip().lower() == str(got).strip().lower():
                continue
        return False
    return True


async def _tool_call_match_one(args, sample) -> float:
    """Agentic-Conversational-Tool-Use / Agentic-Function-Calling-Pivot.

    These pose each expert step as a one-shot decision: given the conversation so
    far and the tool signatures, reproduce the expert's next action. No tool is
    executed and no environment advances, so the reward is a comparison against
    the recorded `expected_action` -- which is why these two components need
    neither a sandbox nor an external service.

    Scoring is deliberately strict and symmetric:
      * expert called a tool  -> the policy must call that tool with those arguments
      * expert replied in text -> the policy must also not call a tool
    Calling a tool when the expert answered directly is a real failure mode
    (spurious tool use), so it scores 0 rather than being ignored.
    """
    expected = (sample.metadata or {}).get("expected_action")
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            expected = None
    signature = expected_action_signature(expected) if expected else None
    if signature is None:
        return 0.0

    emitted = parse_emitted_tool_calls(sample.response)

    if signature["kind"] == "message":
        # Text answers have no single correct wording, so only the decision
        # "do not call a tool" is scored.
        return 1.0 if not emitted else 0.0

    if not emitted:
        return 0.0
    first = emitted[0]
    if first["name"] != signature["name"]:
        return 0.0
    return 1.0 if _arguments_match(signature["arguments"], first["arguments"]) else 0.0


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def _dispatch(scorer):
    """Accept both miles custom-rm contracts (one Sample, or a list of them)."""

    async def entry(args, sample_or_samples, **kwargs):
        if isinstance(sample_or_samples, list):
            return [await scorer(args, s) for s in sample_or_samples]
        return await scorer(args, sample_or_samples)

    return entry


reasoning_gym_reward = _dispatch(_reasoning_gym_one)
structured_output_reward = _dispatch(_structured_output_one)
tool_call_match_reward = _dispatch(_tool_call_match_one)


# --------------------------------------------------------------------------
# preflight support
# --------------------------------------------------------------------------


def build_preflight_probes(label, metadata):
    """(correct, wrong) responses for `verifier_preflight`, supplied by the verifier itself.

    `measure_pass_rate.verifier_preflight` cannot synthesize a correct answer for
    a custom verifier: for maths it can guess `\\boxed{label}`, but "the expert's
    tool call" or "JSON matching this schema" is knowledge only the verifier has.
    Exporting the probe next to the verifier keeps the two from drifting, and
    keeps that dataset-specific knowledge out of the driver.

    The *wrong* probe has to come from here too, not from a generic template.
    A tool-use row whose expert answered in text is graded on "did you avoid
    calling a tool", and the driver's generic wrong answer -- a boxed number --
    calls no tool, so it scores 1.0 and the guard fires on a false positive.
    Whatever defines correctness also defines what counts as wrong.

    Returns None when no probe can be built, which the driver reports rather than
    treating as a failure.
    """
    metadata = metadata or {}
    source = metadata.get("source")

    if source in ("conv-tooluse", "fncall-pivot"):
        signature = expected_action_signature(metadata.get("expected_action"))
        if signature is None:
            return None
        call = lambda name, arguments: (  # noqa: E731
            "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"
        )
        if signature["kind"] == "message":
            # Expert replied in text: calling any tool is the failure mode.
            return "Here is the answer, no tool needed.", call("definitely_not_a_real_tool", {})
        return call(signature["name"], signature["arguments"]), "I will just answer in text instead."

    if source == "structured-outputs":
        schema_str = metadata.get("schema_str")
        if not schema_str:
            return None
        try:
            schema = json.loads(schema_str)
        except json.JSONDecodeError:
            return None
        return json.dumps(_minimal_instance(schema)), "this is not JSON at all"

    if source == "reasoning-gym":
        return f"Working through it.\n\nAnswer: {label}", "Answer: definitely-not-the-answer-42x"

    return None


def _resolve_ref(schema, root):
    """Follow a local $ref. Generated schemas mix `$defs` and Draft-7 `definitions`."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return {}
        if part not in node and part == "$defs" and "definitions" in node:
            part = "definitions"
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def _minimal_instance(schema, root=None, depth=0):
    """Smallest instance satisfying a schema, for the preflight probe only.

    Honours `enum` and `minimum` for every type, not just strings: the real
    schemas in this component constrain numbers too (`printSpeed` has minimum 1,
    `filamentDiameter` is enum [1.75, 2.85]), and a naive 0 fails validation --
    which reads as "the verifier is broken" when it is only the probe that is.
    """
    if not isinstance(schema, dict):
        return {}
    root = schema if root is None else root
    if depth > 12:  # cyclic $ref
        return {}
    if "$ref" in schema:
        schema = _resolve_ref(schema, root)
        if not isinstance(schema, dict):
            return {}
    # An enum pins the value regardless of type, so check it before anything else.
    if schema.get("enum"):
        return schema["enum"][0]
    for key in ("const",):
        if key in schema:
            return schema[key]

    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), kind[0] if kind else None)

    if kind == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {name: _minimal_instance(props.get(name, {}), root, depth + 1) for name in schema.get("required") or []}
    if kind == "array":
        least = schema.get("minItems") or 0
        item_schema = schema.get("items") or {}
        return [_minimal_instance(item_schema, root, depth + 1) for _ in range(least)]
    if kind == "string":
        fmt = schema.get("format")
        if fmt == "date-time":
            return "2026-01-01T00:00:00Z"
        if fmt == "date":
            return "2026-01-01"
        return "x" * max(1, int(schema.get("minLength") or 1))
    if kind in ("integer", "number"):
        low = schema.get("minimum", schema.get("exclusiveMinimum"))
        value = low if low is not None else 0
        if schema.get("exclusiveMinimum") is not None and low is not None:
            value = low + 1
        high = schema.get("maximum")
        if high is not None and value > high:
            value = high
        return int(value) if kind == "integer" else value
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return "x"
