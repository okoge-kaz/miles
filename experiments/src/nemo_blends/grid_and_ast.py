"""Two more verifiers: ARC-AGI grid matching, and BFCL's AST check.

Both are deterministic and need no sandbox, no judge and no second model.

`arc_agi_reward`   nvidia/Nemotron-RL-ARC-AGI-v1 (transductive)
`bfcl_ast_reward`  gorilla-llm/Berkeley-Function-Calling-Leaderboard, AST categories

The BFCL *exec_* categories are excluded on purpose: they call live APIs. The AST
categories -- simple, multiple, parallel, live_* -- are the majority and are
graded by comparing the emitted call against a set of allowed values, which is
the same thing `tool_call_match_reward` does for the Nemotron pivot sets.
"""

import ast
import json
import re

from experiments.src.nemo_blends.rewards import parse_emitted_tool_calls

__all__ = ["arc_agi_reward", "bfcl_ast_reward"]


# --------------------------------------------------------------------------
# ARC-AGI
# --------------------------------------------------------------------------

_BOXED = re.compile(r"\\boxed\{(.*)\}", re.S)


def parse_grid(text: str):
    """The grid the response ended with, as a list of lists of ints.

    ARC answers are exact: a single wrong cell is a wrong answer, so there is no
    tolerance anywhere in here. Accepts the `\\boxed{...}` form the prompt asks
    for, and falls back to the last bracketed block for a model that forgets the
    wrapper but still emits the grid.
    """
    if not text:
        return None
    candidates = []
    boxed = _BOXED.findall(text)
    if boxed:
        candidates.append(boxed[-1])
    # Last balanced [[...]] span, scanned from the end.
    depth = 0
    end = None
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == "]":
            if depth == 0:
                end = i + 1
            depth += 1
        elif c == "[":
            depth -= 1
            if depth == 0 and end is not None:
                candidates.append(text[i:end])
                break
    for raw in candidates:
        raw = raw.strip()
        for parse in (json.loads, ast.literal_eval):
            try:
                grid = parse(raw)
            except (ValueError, SyntaxError):
                continue
            if (
                isinstance(grid, list)
                and grid
                and all(isinstance(row, list) and row and all(isinstance(v, int) for v in row) for row in grid)
            ):
                return grid
    return None


def _arc_one(sample) -> float:
    expected = (sample.metadata or {}).get("expected_output")
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(expected, list) or not expected:
        return 0.0
    return 1.0 if parse_grid(sample.response) == expected else 0.0


async def arc_agi_reward(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_arc_one(s) for s in sample_or_samples]
    return _arc_one(sample_or_samples)


# --------------------------------------------------------------------------
# BFCL AST
# --------------------------------------------------------------------------


def _value_allowed(actual, allowed) -> bool:
    """BFCL lists every acceptable value per parameter, so this is membership,
    not equality. An empty string in the list marks the parameter optional."""
    if not isinstance(allowed, list):
        allowed = [allowed]
    for want in allowed:
        if actual == want:
            return True
        # The two sides disagree on int-vs-str and on case for enums.
        if isinstance(want, (str, int, float, bool)) and isinstance(actual, (str, int, float, bool)):
            if str(want).strip().lower() == str(actual).strip().lower():
                return True
    return False


def _call_matches(call, truth_entry) -> bool:
    """One emitted call against one ground-truth entry {fn_name: {param: [values]}}."""
    if not isinstance(truth_entry, dict) or len(truth_entry) != 1:
        return False
    (fn_name, params), = truth_entry.items()
    if call.get("name") != fn_name:
        return False
    args = call.get("arguments") or {}

    for param, allowed in (params or {}).items():
        optional = isinstance(allowed, list) and "" in allowed
        if param not in args:
            # Absent is fine only when the empty string marks it optional.
            if optional:
                continue
            return False
        if not _value_allowed(args[param], allowed):
            return False
    # An argument the ground truth never mentions changes the call's meaning.
    return not (set(args) - set(params or {}))


def _bfcl_one(sample) -> float:
    """1.0 when the emitted calls match the ground-truth set.

    Order-insensitive: the parallel categories expect several calls and BFCL does
    not fix their order. Count must match too -- emitting an extra call is a
    different behaviour, not a superset of the right one.
    """
    truth = (sample.metadata or {}).get("ground_truth")
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(truth, list):
        return 0.0

    emitted = parse_emitted_tool_calls(sample.response)

    # The irrelevance/relevance categories ship an empty ground truth: the right
    # behaviour is to call nothing.
    if not truth:
        return 1.0 if not emitted else 0.0
    if len(emitted) != len(truth):
        return 0.0

    remaining = list(emitted)
    for entry in truth:
        for i, call in enumerate(remaining):
            if _call_matches(call, entry):
                remaining.pop(i)
                break
        else:
            return 0.0
    return 1.0


async def bfcl_ast_reward(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_bfcl_one(s) for s in sample_or_samples]
    return _bfcl_one(sample_or_samples)


def build_preflight_probes(label, metadata):
    """Correct/wrong probes, taken from whichever ground truth the row carries."""
    metadata = metadata or {}

    expected = metadata.get("expected_output")
    if expected:
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                return None
        return f"The rule maps it as follows.\n\n\\boxed{{{json.dumps(expected)}}}", "\\boxed{[[0]]}"

    truth = metadata.get("ground_truth")
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except json.JSONDecodeError:
            return None
    if isinstance(truth, list):
        if not truth:
            return "No function is relevant here.", '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
        calls = []
        for entry in truth:
            (fn_name, params), = entry.items()
            args = {k: (v[0] if isinstance(v, list) and v else v) for k, v in (params or {}).items()}
            args = {k: v for k, v in args.items() if v != ""}
            calls.append("<tool_call>\n" + json.dumps({"name": fn_name, "arguments": args}) + "\n</tool_call>")
        return "\n".join(calls), "I will answer in prose instead."
    return None
