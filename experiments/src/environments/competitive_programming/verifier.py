"""Sandboxed Python environment verifier for competitive-programming RL rows.

Generated code is untrusted. The default backend therefore requires Bubblewrap
and util-linux ``unshare``, isolates the network/process namespaces, exposes only
the Python runtime read-only, and gives each execution a fresh writable directory. Set
``CODE_EXEC_SANDBOX=process`` only on an already-isolated disposable worker.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import weakref
from pathlib import Path
from typing import Any

__all__ = ["build_preflight_probes", "code_exec_reward", "extract_code", "run_tests"]

DEFAULT_TIMEOUT_S = float(os.environ.get("CODE_EXEC_TIMEOUT", "10"))
DEFAULT_MEMORY_GB = float(os.environ.get("CODE_EXEC_MEMORY_GB", "4"))
MAX_TESTS = int(os.environ.get("CODE_EXEC_MAX_TESTS", "50"))
MAX_OUTPUT_BYTES = int(os.environ.get("CODE_EXEC_MAX_OUTPUT_BYTES", str(16 * 1024 * 1024)))
SANDBOX_BACKEND = os.environ.get("CODE_EXEC_SANDBOX", "bubblewrap")
CONCURRENCY = int(os.environ.get("CODE_EXEC_CONCURRENCY", "4"))

# Miles may invoke a custom reward once per sample instead of passing a whole
# reward batch.  A semaphore created inside ``code_exec_reward`` would then cap
# only that single call and allow every sample to launch a subprocess at once.
# Keep one limiter per event loop so both reward contracts share the same
# rollout-manager execution budget without binding a semaphore across test loops.
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(response: str) -> str:
    """Extract the last Python code block, or accept parseable bare code."""
    blocks = _FENCE.findall(response or "")
    if blocks:
        return blocks[-1].strip()
    text = (response or "").strip()
    try:
        ast.parse(text)
    except SyntaxError:
        return ""
    return text


def _limited_command(command: list[str], *, memory_gb: float, timeout: float) -> list[str]:
    """Apply limits without an unsafe ``preexec_fn`` in the threaded scorer."""
    executable = shutil.which("prlimit")
    if executable is None:
        raise RuntimeError("code execution requires util-linux prlimit")
    memory_bytes = int(memory_gb * 1024**3)
    cpu_soft = math.ceil(timeout)
    return [
        executable,
        f"--as={memory_bytes}",
        "--core=0",
        f"--fsize={MAX_OUTPUT_BYTES}",
        "--nofile=64",
        f"--cpu={cpu_soft}:{cpu_soft + 1}",
        "--",
        *command,
    ]


def _bubblewrap_command(workdir: Path, script: Path) -> list[str]:
    executable = shutil.which("bwrap")
    unshare = shutil.which("unshare")
    if executable is None or unshare is None:
        raise RuntimeError(
            "code execution requires Bubblewrap and util-linux unshare; install both or explicitly "
            "run a disposable isolated worker with CODE_EXEC_SANDBOX=process"
        )
    # The cluster seccomp profile permits creating a network namespace but denies
    # Bubblewrap's NETLINK_ROUTE request for bringing up loopback. Create the
    # namespace with unshare first (loopback stays down), then let Bubblewrap own
    # the filesystem and remaining namespaces. The inherited namespace has no
    # routable interfaces, so outbound connects fail closed.
    command = [
        unshare,
        "--user",
        "--map-root-user",
        "--net",
        "--fork",
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/home",
        "--setenv",
        "HOME",
        "/home",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
    ]
    for runtime_path in ("/usr", "/usr/local", "/bin", "/lib", "/lib64"):
        if Path(runtime_path).exists():
            command.extend(["--ro-bind", runtime_path, runtime_path])
    command.extend(["--bind", str(workdir), "/work", "--chdir", "/work"])
    # `/usr/bin/python3` is an alternatives symlink through `/etc`, which is
    # deliberately absent from the sandbox. Resolve it before entering bwrap.
    python_path = os.path.realpath(sys.executable)
    command.extend([python_path, f"/work/{script.name}"])
    return command


def _run(source: str, stdin_text: str, timeout: float, memory_gb: float) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix="miles-code-") as temp_dir:
        workdir = Path(temp_dir)
        script = workdir / "solution.py"
        script.write_text(source, encoding="utf-8")
        if SANDBOX_BACKEND == "bubblewrap":
            command = _bubblewrap_command(workdir=workdir, script=script)
            cwd = None
        elif SANDBOX_BACKEND == "process":
            command = [sys.executable, str(script)]
            cwd = workdir
        else:
            raise ValueError(f"unknown CODE_EXEC_SANDBOX backend: {SANDBOX_BACKEND}")
        command = _limited_command(command, memory_gb=memory_gb, timeout=timeout)
        try:
            process = subprocess.run(
                command,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return "", False
        if os.environ.get("CODE_EXEC_DEBUG") == "1" and process.stderr:
            print(process.stderr, file=sys.stderr, end="")
        if len(process.stdout.encode(errors="ignore")) > MAX_OUTPUT_BYTES:
            return "", False
        return process.stdout, process.returncode == 0


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").strip().splitlines()).strip()


def _function_call_harness(source: str, function_name: str, raw_input: str) -> str:
    arguments = []
    for line in str(raw_input or "").splitlines():
        if not line.strip():
            continue
        try:
            arguments.append(ast.literal_eval(line.strip()))
        except (SyntaxError, ValueError):
            arguments.append(line)
    encoded_arguments = json.dumps(arguments)
    return (
        source
        + "\n\nimport inspect as _inspect, json as _json\n"
        + f"_args = _json.loads({encoded_arguments!r})\n"
        + f"_fn = globals().get({function_name!r})\n"
        + "if _fn is None:\n"
        + "    for _value in list(globals().values()):\n"
        + f"        if _inspect.isclass(_value) and hasattr(_value, {function_name!r}):\n"
        + f"            _fn = getattr(_value(), {function_name!r})\n"
        + "            break\n"
        + "if _fn is None:\n"
        + f"    raise SystemExit('no callable named {function_name}')\n"
        + "print(_json.dumps(_fn(*_args)))\n"
    )


def _published_harness(source: str, unit_tests: dict[str, Any]) -> str:
    entry_point = str(unit_tests.get("entry_point") or "").strip()
    test_code = str(unit_tests.get("test_code") or "").strip()
    import_prefix = str(unit_tests.get("import_prefix") or "")
    expression = ast.parse(entry_point, mode="eval").body
    is_function = isinstance(expression, ast.Name)
    is_method = (
        isinstance(expression, ast.Attribute)
        and isinstance(expression.value, ast.Call)
        and isinstance(expression.value.func, ast.Name)
        and not expression.value.args
        and not expression.value.keywords
    )
    if not test_code or not (is_function or is_method):
        raise ValueError(f"unsupported published entry point: {entry_point!r}")
    test_tree = ast.parse(test_code)
    has_check = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check" for node in test_tree.body
    )
    if not has_check:
        raise ValueError("published test harness does not define check(candidate)")
    return (
        import_prefix
        + "\n"
        + source
        + "\n\n"
        + test_code
        + f"\n_candidate = {entry_point}\n"
        + "_check_result = check(_candidate)\n"
        + "if _check_result is False:\n"
        + "    raise AssertionError('published check returned False')\n"
    )


def _outputs_match(stdout: str, expected: Any, *, function_call: bool) -> bool:
    actual = _normalize(stdout)
    wanted = _normalize(str(expected))
    if actual == wanted:
        return True
    if not function_call:
        return False
    try:
        if json.loads(actual) == json.loads(wanted):
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return actual.strip('"') == wanted.strip('"')


def run_tests(
    source: str,
    unit_tests: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> float:
    """Return one only when every selected unit test passes."""
    if unit_tests.get("test_code") or unit_tests.get("entry_point"):
        try:
            program = _published_harness(source, unit_tests)
        except (SyntaxError, ValueError):
            return 0.0
        _, succeeded = _run(program, "", timeout=timeout, memory_gb=memory_gb)
        return 1.0 if succeeded else 0.0
    inputs = unit_tests.get("inputs") or []
    outputs = unit_tests.get("outputs") or []
    function_name = unit_tests.get("fn_name")
    if not inputs or len(inputs) != len(outputs):
        return 0.0
    for raw_input, expected in list(zip(inputs, outputs, strict=True))[:MAX_TESTS]:
        if function_name:
            program = _function_call_harness(source, str(function_name), str(raw_input))
            stdin_text = ""
        else:
            program = source
            stdin_text = str(raw_input)
        stdout, succeeded = _run(program, stdin_text, timeout=timeout, memory_gb=memory_gb)
        if not succeeded or not _outputs_match(stdout, expected, function_call=bool(function_name)):
            return 0.0
    return 1.0


def _score_one(sample: Any) -> float:
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
    return run_tests(source, unit_tests) if source else 0.0


def _execution_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    entry = _LOOP_SEMAPHORES.get(loop)
    if entry is None or entry[0] != CONCURRENCY:
        entry = (CONCURRENCY, asyncio.Semaphore(CONCURRENCY))
        _LOOP_SEMAPHORES[loop] = entry
    return entry[1]


async def code_exec_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Support both Miles custom-reward contracts without blocking its event loop."""
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    semaphore = _execution_semaphore()

    async def score(sample: Any) -> float:
        async with semaphore:
            return await asyncio.to_thread(_score_one, sample)

    rewards = await asyncio.gather(*(score(sample) for sample in samples))
    return rewards if isinstance(sample_or_samples, list) else rewards[0]


def build_preflight_probes(label: Any, metadata: dict[str, Any]) -> tuple[str, str] | None:
    tests = metadata.get("unit_tests") if isinstance(metadata, dict) else None
    outputs = tests.get("outputs") if isinstance(tests, dict) else None
    if not isinstance(tests, dict) or tests.get("fn_name") or not outputs or len(outputs) != 1:
        return None
    expected = str(outputs[0])
    correct = f"```python\nprint({expected!r}, end='')\n```"
    wrong = "```python\nprint('definitely-not-the-answer')\n```"
    return correct, wrong
