"""Run a generated Python solution against its unit tests, locally.

Used by `nvidia/Nemotron-RL-coding-competitive_coding` and by LiveCodeBench --
the two share this harness because they share the problem shape.

No container. Execution is a `subprocess` with a wall-clock timeout and an
address-space rlimit, the same weak-isolation pattern
`examples/retool_v2/tool_sandbox.py` already uses for ReTool. That is not a
security boundary and is not claimed to be one; it is enough for grading code
the policy wrote against tests we supply, on a node we already trust.

Two problem shapes appear in the data and both are handled:

  * **function-call** -- `verifier_metadata.unit_tests` has `fn_name`, and each
    input is the literal argument list (`"[3, 2, 2]\\n2"` means two arguments).
    LeetCode-style; the solution defines a class or a bare function.
  * **stdin/stdout** -- no `fn_name`; the input is fed to the program's stdin and
    the whole stdout is compared. Codeforces/AtCoder-style.

Scoring is all-or-nothing across the test list. Competitive-programming problems
are judged that way, and partial credit would reward a solution that special-cases
the samples.
"""

import ast
import json
import os
import re
import resource
import subprocess
import sys
import tempfile

__all__ = ["code_exec_reward", "extract_code", "run_tests"]

# Per test case. Long enough for an O(n log n) solution on the biggest inputs in
# these sets, short enough that an accidental infinite loop costs one slot rather
# than a rollout.
DEFAULT_TIMEOUT_S = float(os.environ.get("CODE_EXEC_TIMEOUT", "10"))
DEFAULT_MEMORY_GB = float(os.environ.get("CODE_EXEC_MEMORY_GB", "4"))
# Stop at the first failure, but cap total work for a solution that passes
# thousands of cases slowly.
MAX_TESTS = int(os.environ.get("CODE_EXEC_MAX_TESTS", "40"))

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(response: str) -> str:
    """The last fenced block, or the whole response if it is bare code.

    Last, not first: a model that explains an approach and then writes the final
    program would otherwise be graded on its illustration.
    """
    blocks = _FENCE.findall(response or "")
    if blocks:
        return blocks[-1].strip()
    text = (response or "").strip()
    # No fence: accept it only if it parses, so prose is not run as a program.
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return ""


def _limits(memory_gb: float):
    def apply():
        nbytes = int(memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        # No core dumps: a crashing solution would otherwise write GBs onto lustre.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _run(source: str, stdin_text: str, timeout: float, memory_gb: float):
    """Run one program. Returns (stdout, ok)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                preexec_fn=_limits(memory_gb),
                # An inherited environment lets the solution see cluster
                # credentials it has no business reading.
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return "", False
        except Exception:
            return "", False
    return proc.stdout, proc.returncode == 0


def _normalize(text: str) -> str:
    """Compare the way a judge does: trailing whitespace and blank tails are not
    wrong answers."""
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines()).strip()


def _function_call_harness(source: str, fn_name: str, raw_input: str) -> str:
    """Wrap a solution so its function is called with the literal arguments.

    Each line of the input is one argument, written as a Python literal. The
    solution may define `fn_name` at module level or as a method on a class
    (LeetCode's `Solution`), so both are looked up.
    """
    args = []
    for line in (raw_input or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            args.append(ast.literal_eval(line))
        except (ValueError, SyntaxError):
            args.append(line)
    return (
        source
        + "\n\n"
        + "import json as _json, inspect as _inspect\n"
        + f"_args = _json.loads({json.dumps(json.dumps(args))})\n"
        + f"_fn = globals().get({fn_name!r})\n"
        + "if _fn is None:\n"
        + "    for _v in list(globals().values()):\n"
        + "        if _inspect.isclass(_v) and hasattr(_v, %r):\n" % fn_name
        + f"            _fn = getattr(_v(), {fn_name!r}); break\n"
        + "if _fn is None:\n"
        + f"    raise SystemExit('no callable named {fn_name}')\n"
        + "print(_json.dumps(_fn(*_args)))\n"
    )


def run_tests(source: str, unit_tests: dict, timeout=DEFAULT_TIMEOUT_S, memory_gb=DEFAULT_MEMORY_GB) -> float:
    """1.0 only if every case matches. Returns 0.0 on the first mismatch."""
    inputs = unit_tests.get("inputs") or []
    outputs = unit_tests.get("outputs") or []
    fn_name = unit_tests.get("fn_name")
    if not inputs or len(inputs) != len(outputs):
        return 0.0

    for raw_in, expected in list(zip(inputs, outputs, strict=True))[:MAX_TESTS]:
        if fn_name:
            program = _function_call_harness(source, fn_name, str(raw_in))
            stdin_text = ""
        else:
            program = source
            stdin_text = str(raw_in)

        stdout, ok = _run(program, stdin_text, timeout, memory_gb)
        if not ok:
            return 0.0

        got, want = _normalize(stdout), _normalize(str(expected))
        if got == want:
            continue
        # The function-call path prints JSON, so compare structurally before
        # giving up: [1, 2] and [1,2] are the same answer.
        if fn_name:
            try:
                if json.loads(got) == json.loads(want):
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            # A scalar result is often written bare in the expected output.
            if got.strip('"') == want.strip('"'):
                continue
        return 0.0
    return 1.0


def _score_one(sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    unit_tests = metadata.get("unit_tests")
    if isinstance(unit_tests, str):
        try:
            unit_tests = json.loads(unit_tests)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(unit_tests, dict):
        return 0.0

    source = extract_code(sample.response)
    if not source:
        return 0.0
    return run_tests(source, unit_tests)


async def code_exec_reward(args, sample_or_samples, **kwargs):
    """Both miles custom-rm contracts (a list from `batched_async_rm`, one Sample
    from `async_rm`)."""
    if isinstance(sample_or_samples, list):
        return [_score_one(s) for s in sample_or_samples]
    return _score_one(sample_or_samples)


def build_preflight_probes(label, metadata):
    """A correct program for this row's tests, and one that is wrong.

    Only synthesizable for the stdin/stdout shape, where echoing the first
    expected output passes the first case. That is enough to prove the harness
    runs code and compares output; the function-call shape returns None and the
    driver reports that rather than failing.
    """
    metadata = metadata or {}
    tests = metadata.get("unit_tests")
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            return None
    if not isinstance(tests, dict) or tests.get("fn_name"):
        return None
    outputs = tests.get("outputs") or []
    if not outputs:
        return None
    # Cheating on purpose: the probe tests the harness, not a policy.
    correct = "```python\nimport sys\nprint(%r, end='')\n```" % str(outputs[0])
    wrong = "```python\nprint('definitely-not-the-answer')\n```"
    return correct, wrong
