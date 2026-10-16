import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager, suppress


def server_command(config, engine_index):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.server_port + 10 * engine_index),
        "--tp-size",
        str(config.tensor_parallel_size),
        "--dp-size",
        "1",
        "--base-gpu-id",
        str(engine_index * config.tensor_parallel_size),
        "--context-length",
        str(config.max_context_length),
        "--random-seed",
        # RL assigns different seeds to independently launched inference engines.
        str(config.seed + engine_index),
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--max-running-requests",
        str(config.max_running_requests),
        "--log-level",
        "info",
    ]


def wait_for_servers(processes, config):
    deadline = time.monotonic() + config.startup_timeout
    base_urls = [f"http://127.0.0.1:{config.server_port + 10 * index}" for index in range(config.num_engines)]
    pending = set(range(config.num_engines))
    while time.monotonic() < deadline:
        for index, process in enumerate(processes):
            if process.poll() is not None:
                raise RuntimeError(f"SGLang engine {index} exited during startup: {process.returncode}")
        for index in sorted(pending):
            try:
                with urllib.request.urlopen(base_urls[index] + "/health", timeout=2):
                    pass
                with urllib.request.urlopen(base_urls[index] + "/get_model_info", timeout=2) as response:
                    info = json.load(response)
                if info["model_path"] != config.model_path:
                    raise ValueError(f"SGLang engine {index} is serving a different model")
                pending.remove(index)
            except (urllib.error.URLError, TimeoutError):
                continue
        if not pending:
            return base_urls
        time.sleep(2)
    raise TimeoutError("SGLang startup timed out")


def stop_servers(processes):
    for process in processes:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@contextmanager
def serve(config, directory):
    # Check every endpoint before starting any model process.
    with ExitStack() as probes:
        for index in range(config.num_engines):
            probe = probes.enter_context(socket.socket())
            probe.bind(("127.0.0.1", config.server_port + 10 * index))
    processes = []
    with ExitStack() as logs:
        try:
            for index in range(config.num_engines):
                command = server_command(config, index)
                log_path = directory / f"sglang-engine-{index}.log"
                print(f"SGLang engine {index}, log={log_path}: {shlex.join(command)}", flush=True)
                log = logs.enter_context(log_path.open("a"))
                processes.append(
                    subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                )
            yield wait_for_servers(processes, config)
        finally:
            stop_servers(processes)
