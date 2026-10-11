import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).with_name("sources.lock.json")
SERVICES = {
    "instruction_following": ("resources_servers", "instruction_following"),
    "ifbench": ("resources_servers", "ifbench"),
    "reasoning_gym": ("resources_servers", "reasoning_gym"),
    "calendar": ("resources_servers", "calendar"),
    "workplace_resources": ("resources_servers", "workplace_assistant"),
    "workplace": ("responses_api_agents", "simple_agent"),
    "tau2": ("responses_api_agents", "tau2"),
    "user_model": ("responses_api_models", "openai_model"),
    "judge_model": ("responses_api_models", "vllm_model"),
    "mcqa": ("resources_servers", "mcqa"),
    "hotpotqa_qa": ("resources_servers", "hotpotqa_qa"),
    "structured_outputs": ("resources_servers", "structured_outputs"),
    "code_gen": ("resources_servers", "code_gen"),
    "competitive_coding_challenges": ("resources_servers", "competitive_coding_challenges"),
    "math_with_judge": ("resources_servers", "math_with_judge"),
    "equivalence_llm_judge": ("resources_servers", "equivalence_llm_judge"),
    "multichallenge": ("resources_servers", "multichallenge"),
}
JUDGE_SERVICES = {"math_with_judge", "equivalence_llm_judge", "multichallenge"}


def model_config(path: str, *, chat_adapter=False) -> dict:
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    url_key, model_key, key_key = (
        ("base_url", "model", "api_key") if chat_adapter else ("openai_base_url", "openai_model", "openai_api_key")
    )
    if not config.get(url_key) or not config.get(model_key):
        raise ValueError(f"Model config requires native {url_key} and {model_key} fields")
    if config.get(key_key) and config_path.stat().st_mode & 0o077:
        raise ValueError("Model config containing an API key must be owner-only (chmod 600)")
    defaults = {key_key: "unused"}
    if chat_adapter:
        defaults.update(
            return_token_id_information=False,
            uses_reasoning_parser=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    else:
        defaults.update(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            upstream_max_num_tries=1,
            upstream_request_timeout_seconds=120,
            upstream_retry_policy={"max_attempts": 1},
        )
    return {**defaults, **config}


def configuration(
    host: str,
    port_base: int,
    services: list[str],
    user_endpoint: str | None,
    max_steps: int,
    *,
    judge_endpoint=None,
    code_test_file=None,
    code_shared_dir=None,
    code_timeout=10,
    code_concurrency=4,
) -> tuple[dict, dict]:
    names = list(
        dict.fromkeys(
            [
                *services,
                *(["workplace_resources"] if "workplace" in services else []),
                *(["user_model"] if "tau2" in services else []),
                *(["judge_model"] if JUDGE_SERVICES.intersection(services) else []),
            ]
        )
    )
    config = {
        "dry_run": False,
        "observability_enabled": True,
        "head_server": {"host": host, "port": port_base},
        "policy_model": {
            "responses_api_models": {"miles_policy": {"host": host, "port": port_base, "entrypoint": "unused"}}
        },
    }
    urls = {}
    for index, name in enumerate(names, 1):
        kind, directory = SERVICES[name]
        fields = {"host": host, "port": port_base + index, "entrypoint": "app.py", "num_workers": 1}
        if name in {"workplace", "tau2"}:
            fields.update(model_server={"type": "responses_api_models", "name": "policy_model"}, max_steps=max_steps)
        if name == "workplace":
            fields["resources_server"] = {"type": "resources_servers", "name": "workplace_resources"}
        elif name == "tau2":
            fields.update(
                user_model_server={"type": "responses_api_models", "name": "user_model"},
                user_llm_args={"temperature": 0.7, "max_tokens": 2048},
                malformed_tool_call_max_retries=0,
            )
        elif name == "user_model":
            if not user_endpoint:
                raise ValueError("Tau2 requires --user-model-config for the frozen, separate simulator")
            fields.update(model_config(user_endpoint))
        elif name == "ifbench":
            fields["num_processes"] = 2
        elif name == "code_gen":
            fields.update(num_processes=code_concurrency, unit_test_timeout_secs=code_timeout, debug=False)
        elif name == "competitive_coding_challenges":
            if not code_test_file or not code_shared_dir:
                raise ValueError("C++ requires --code-test-file with native NeMo Gym test metadata")
            fields.update(
                test_file=str(Path(code_test_file).resolve()),
                shared_dir=str(code_shared_dir),
                local_compile_dir=None,
                num_parallel_requests=code_concurrency,
                test_batch_size=code_concurrency,
                reward_mode="binary",
            )
        elif name == "judge_model":
            if not judge_endpoint:
                raise ValueError("LLM judging requires --judge-model-config for a separate frozen judge")
            fields.update(model_config(judge_endpoint, chat_adapter=True))
        elif name in JUDGE_SERVICES:
            fields.update(
                judge_model_server={"type": "responses_api_models", "name": "judge_model"},
                judge_responses_create_params={"input": [], "max_output_tokens": 2048, "temperature": 0},
            )
        config[name] = {kind: {directory: fields}}
        urls[name] = f"http://{host}:{port_base + index}"
    return config, urls


def check_install(root: Path) -> dict:
    provenance = json.loads((root / "miles-runtime.json").read_text())
    if provenance["sources"] != json.loads(SOURCES.read_text()):
        raise ValueError("NeMo Gym runtime source lock differs; run setup for the new lock")
    patches = {
        patch.name: hashlib.sha256(patch.read_bytes()).hexdigest()
        for patch in (SOURCES.parent.parent / "patches").glob("*.patch")
    }
    if provenance.get("patches_sha256") != patches:
        raise ValueError("NeMo Gym patch contents changed; rebuild the runtime")
    if (
        provenance.get("requirements_sha256")
        != hashlib.sha256(SOURCES.with_name("requirements.lock.txt").read_bytes()).hexdigest()
    ):
        raise ValueError("NeMo Gym dependency lock changed; rebuild the runtime")
    expected = {
        line.removeprefix("+++ b/")
        for patch in (SOURCES.parent.parent / "patches").glob("*.patch")
        for line in patch.read_text().splitlines()
        if line.startswith("+++ b/")
    }
    if set(provenance["patched_sha256"]) != expected:
        raise ValueError("NeMo Gym runtime patches changed; run setup again")
    for name, digest in provenance["patched_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"NeMo Gym verifier error patch differs: {name}")
    return provenance


def wait_ready(processes: list, urls: list[str], timeout: float):
    deadline = time.monotonic() + timeout
    pending = set(urls)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while pending and time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A NeMo Gym service exited; see service logs")
        for url in tuple(pending):
            try:
                with opener.open(url + "/health", timeout=1) as response:
                    if response.status == 200:
                        pending.remove(url)
            except OSError:
                pass
        if pending:
            time.sleep(1)
    if pending:
        raise TimeoutError(f"NeMo Gym readiness timeout: {sorted(pending)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run upstream NeMo Gym CPU services inside the allocated job")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--host", required=True, help="Compute-node IP reachable from the Miles workers")
    parser.add_argument("--port-base", type=int, default=30190)
    parser.add_argument(
        "--services",
        nargs="+",
        choices=[name for name in SERVICES if name not in {"workplace_resources", "user_model", "judge_model"}],
        default=[
            "instruction_following",
            "tau2",
            "code_gen",
            "mcqa",
        ],
    )
    parser.add_argument("--user-model-config")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--judge-model-config")
    parser.add_argument("--code-test-file", type=Path)
    parser.add_argument("--code-timeout", type=int, default=10)
    parser.add_argument("--code-concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.code_timeout <= 0 or args.code_concurrency <= 0:
        parser.error("Code timeout and concurrency must be positive")
    return args


def check_code_commands(services):
    if {"code_gen", "competitive_coding_challenges"}.intersection(services):
        commands = ["bwrap", "unshare", "mount", "prlimit"]
        if "competitive_coding_challenges" in services:
            commands += ["g++", "ld", "as", "timeout"]
        missing = [name for name in commands if shutil.which(name) is None]
        if missing:
            raise RuntimeError(f"Missing NeMo Gym sandbox commands: {missing}")


def main():
    args = parse_args()
    root = args.root.resolve()
    provenance = check_install(root)
    check_code_commands(args.services)
    shared = (args.state_dir / "code").resolve()
    shared.mkdir(parents=True, exist_ok=True)
    config, urls = configuration(
        args.host,
        args.port_base,
        args.services,
        args.user_model_config,
        args.max_steps,
        judge_endpoint=args.judge_model_config,
        code_test_file=args.code_test_file,
        code_shared_dir=shared,
        code_timeout=args.code_timeout,
        code_concurrency=args.code_concurrency,
    )
    if "equivalence_llm_judge" in config:
        config["equivalence_llm_judge"]["resources_servers"]["equivalence_llm_judge"][
            "judge_prompt_template_fpath"
        ] = str(root / "resources_servers/equivalence_llm_judge/prompt_templates/equivalence_llm_judge.txt")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    config["model_call_capture_dir"] = str((args.state_dir / "model_calls").resolve())
    config["cache_dir"] = str((args.state_dir / "cache").resolve())
    config["results_dir"] = str((args.state_dir / "results").resolve())
    manifest_path = args.state_dir / "endpoints.json"
    if manifest_path.exists():
        raise FileExistsError(f"Choose a fresh state directory: {manifest_path}")
    repo = Path(__file__).resolve().parents[4]
    env = {
        **os.environ,
        "NEMO_GYM_CONFIG_DICT": json.dumps(config),
        "PYTHONPATH": f"{repo}:{root}",
        "NLTK_DATA": str(root / "nltk_data"),
        "NEMO_GYM_TAU2_BENCH_REF": provenance["sources"]["tau_commit"],
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "NEMO_GYM_ROOT": str(root),
        "NEMO_GYM_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "NEMO_GYM_CODE_SHARED_DIR": str(shared),
    }
    processes, logs = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        commands = [
            (
                "policy_model",
                [
                    sys.executable,
                    "-m",
                    "experiments.src.nemo_gym.bridge.policy_server",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port_base),
                ],
            )
        ]
        commands += [
            (name, [sys.executable, str(root / kind / directory / "app.py")])
            for name in urls
            for kind, directory in [SERVICES[name]]
        ]
        for name, command in commands:
            log = (args.state_dir / f"{name}.log").open("w")
            logs.append(log)
            server_config = (
                {**config, "observability_enabled": False} if name in {"user_model", "judge_model"} else config
            )
            server_env = {**env, "NEMO_GYM_CONFIG_PATH": name, "NEMO_GYM_CONFIG_DICT": json.dumps(server_config)}
            processes.append(
                subprocess.Popen(
                    command, cwd=root, env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            )
        bridge = f"http://{args.host}:{args.port_base}"
        wait_ready(processes, [bridge, *urls.values()], 300)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "services": {key: urls[key] for key in args.services},
                    "policy_bridge": bridge,
                    "sources": provenance["sources"],
                    "verifier_errors": "raise",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"NEMO_GYM_ENDPOINTS={manifest_path.resolve()}", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError("NeMo Gym service stopped; see service logs")
    except KeyboardInterrupt:
        pass
    finally:
        manifest_path.unlink(missing_ok=True)
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
