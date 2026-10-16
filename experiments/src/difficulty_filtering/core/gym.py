import argparse
import os
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path


def verifier_command(config, state):
    return [
        "/opt/nemo-gym/.venv/bin/python",
        "-m",
        "experiments.src.nemo_gym.runtime.serve",
        "--root",
        "/opt/nemo-gym",
        "--state-dir",
        str(state),
        "--host",
        "127.0.0.1",
        "--port-base",
        str(config.gym_port),
        "--services",
        config.reward_backend,
        "--code-concurrency",
        str(config.code_concurrency),
    ]


@contextmanager
def serve_verifier(config, directory):
    if config.reward_backend == "deepscaler":
        yield
        return
    from experiments.src.nemo_gym.bridge.client import endpoints as cached_endpoints

    cached_endpoints.cache_clear()
    if os.environ.get("NEMO_GYM_ENDPOINTS"):
        try:
            if config.reward_backend not in cached_endpoints()["services"]:
                raise ValueError(f"External NeMo Gym has no {config.reward_backend} service")
            print(f"Using external NeMo Gym: {os.environ['NEMO_GYM_ENDPOINTS']}", flush=True)
            yield
        finally:
            cached_endpoints.cache_clear()
        return
    state = directory / "gym"
    state.mkdir(exist_ok=True)
    endpoints = state / "endpoints.json"
    endpoints.unlink(missing_ok=True)
    command = verifier_command(config, state)
    previous = os.environ.get("NEMO_GYM_ENDPOINTS")
    with (directory / "gym.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + config.startup_timeout
            while not endpoints.is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"NeMo Gym exited: inspect {directory / 'gym.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("NeMo Gym startup timed out")
                time.sleep(1)
            os.environ["NEMO_GYM_ENDPOINTS"] = str(endpoints)
            yield
        finally:
            cached_endpoints.cache_clear()
            if previous is None:
                os.environ.pop("NEMO_GYM_ENDPOINTS", None)
            else:
                os.environ["NEMO_GYM_ENDPOINTS"] = previous
            # The native runtime handles SIGTERM and reaps its own service processes.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def verifier_probe(backend):
    if backend == "code_gen":
        row = {
            "prompt": "Double the input",
            "label": "code",
            "metadata": {
                "verifier": "python_code",
                "unit_tests": {"inputs": ["2\n"], "outputs": ["4\n"]},
            },
        }
        texts = ["```python\nprint(int(input()) * 2)\n```", "```python\nprint(0)\n```"]
        return row, texts, [1.0, 0.0]
    if backend == "mcqa":
        return (
            {"prompt": "Choose A", "label": "A", "metadata": {"verifier": "mcqa_regex"}},
            ["<think>Answer: B</think>Answer: A", "Answer: B"],
            [1.0, 0.0],
        )
    if backend == "instruction_following":
        row = {
            "prompt": "Say orchid and lily",
            "label": "if",
            "metadata": {
                "verifier": "ifeval_g",
                "prompt_text": "Say orchid and lily",
                "instruction_id_list": ["keywords:existence", "keywords:existence"],
                "kwargs": [{"keywords": ["orchid"]}, {"keywords": ["lily"]}],
            },
        }
        return row, ["orchid lily", "orchid", "rose"], [1.0, 0.5, 0.0]
    return (
        {"prompt": "Compute 3 + 4.", "label": "7"},
        ["<think>3 + 4 = 7</think>\\boxed{7}", "<think>work</think>\\boxed{8}"],
        [1.0, 0.0],
    )


def main():
    from experiments.src.difficulty_filtering.core.config import load_config

    parser = argparse.ArgumentParser(description="Start the filtering verifier inside a CPU-only container")
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.reward_backend == "deepscaler":
        parser.error("DeepScaler does not need a NeMo Gym service")
    command = verifier_command(config, args.state_dir)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
