import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import HTTPException


def fail(kind, *, retryable=True):
    raise HTTPException(503, detail={"failure_kind": kind, "retryable": retryable})


def sandbox_command(command, *, writable, readonly=(), memory_gb=4):
    for name in ("bwrap", "unshare", "mount", "prlimit"):
        if shutil.which(name) is None:
            fail("sandbox_unavailable")
    args = [
        "prlimit",
        f"--as={int(memory_gb * 1024**3)}",
        "--core=0",
        "--fsize=16777216",
        "--nofile=128",
        "--",
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--fork",
        "/bin/sh",
        "-c",
        '"$1" --make-rprivate / && shift && exec "$@"',
        "nemo-code-sandbox",
        shutil.which("mount"),
        shutil.which("bwrap"),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
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
        "PATH",
        "/usr/local/bin:/usr/bin:/bin",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
    ]
    for path in dict.fromkeys(["/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", *map(str, readonly)]):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    for path in writable:
        args += ["--bind", str(path), str(path)]
    return [*args, "--", *map(str, command)]


def execute(command, *, writable, readonly=(), timeout, timeout_kind, memory_gb=4):
    args = sandbox_command(command, writable=writable, readonly=readonly, memory_gb=memory_gb)
    # File-backed output bounds memory usage; RLIMIT_FSIZE also bounds child writes.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                args,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            )
        except OSError:
            fail("sandbox_start_failed")
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            fail(timeout_kind, retryable=False)
        stdout.seek(0)
        stderr.seek(0)
        return (
            process.returncode,
            stdout.read(16 * 1024**2).decode(errors="replace"),
            stderr.read(16 * 1024**2).decode(errors="replace"),
        )


def _check_python(sample, code, timeout):
    root = Path(os.environ["NEMO_GYM_ROOT"])
    grader = root / "resources_servers/code_gen/lcb_integration/testing_util.py"
    runner = Path(__file__).with_name("worker.py")
    interpreter_link = Path(sys.executable).readlink() if Path(sys.executable).is_symlink() else Path(sys.executable)
    interpreter_root = (Path(sys.executable).parent / interpreter_link).absolute().parent.parent
    tests = json.loads(sample["input_output"])
    with tempfile.TemporaryDirectory(prefix="nemo-python-") as directory:
        work = Path(directory)
        request = work / "request.json"
        response = work / "result.json"
        request.write_text(json.dumps({"tests": tests, "code": code, "timeout": timeout}))
        status, _, _ = execute(
            [sys.executable, "-I", runner, grader, request, response],
            writable=[work],
            readonly=[sys.prefix, sys.base_prefix, interpreter_root, runner, grader],
            timeout=(timeout + 1) * len(tests["inputs"]) + 10,
            timeout_kind="grader_deadline",
        )
        if not response.exists():
            fail("sandbox_start_failed" if not (work / "started").exists() else "grader_process_failed")
        result = json.loads(response.read_text())
        if result.get("failure_kind"):
            fail(result["failure_kind"], retryable=False)
        if status:
            fail("grader_process_failed")
        results, metadata = result["results"], result["metadata"]
        if not results or any(value not in (True, False, -2, -3, -4) for value in results):
            fail("invalid_grader_result")
        if metadata.get("error_code") == -3 or -3 in results:
            fail("execution_time_limit", retryable=False)
        metadata["scoring_reason"] = (
            "passed"
            if all(value is True for value in results)
            else "wrong_answer"
            if metadata.get("error_code") == -2
            else "candidate_error"
        )
        return results, metadata


async def check_python(sample, code, timeout, debug=False):
    return await asyncio.to_thread(_check_python, sample, code, timeout)


async def execute_shell(code, *, timeout, language, work_dir=None):
    if language != "shell":
        fail("unsupported_sandbox_language", retryable=False)
    root = Path(os.environ["NEMO_GYM_CODE_SHARED_DIR"]).resolve()
    writable = []
    if work_dir is not None:
        work = Path(work_dir).resolve()
        if not work.is_relative_to(root) or work == root:
            fail("invalid_sandbox_work_dir", retryable=False)
        writable = [work]
    kind = "compilation_deadline" if "./compile.sh" in code else "execution_deadline"
    status, stdout, stderr = await asyncio.to_thread(
        execute,
        ["/bin/bash", "-c", code],
        writable=writable,
        timeout=timeout,
        timeout_kind=kind,
    )
    if status in (124, 137, 143) or "time limit exceeded" in stderr.lower():
        fail("compilation_deadline" if kind == "compilation_deadline" else "execution_time_limit", retryable=False)
    if status < 0:
        fail("sandbox_process_failed")
    if status and any(marker in stderr for marker in ("bwrap:", "unshare:", "prlimit:")):
        fail("sandbox_start_failed")
    return {"process_status": "completed" if status == 0 else "error", "stdout": stdout, "stderr": stderr}, None
