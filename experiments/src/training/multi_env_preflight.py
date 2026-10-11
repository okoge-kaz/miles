import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from miles.utils.dataset_mixture_config import read_prompt_config


def read_sources(value: str | Path, *, files: list[str] | None = None) -> list[tuple[dict, dict]]:
    if value:
        config, directory = read_prompt_config(value)
        sources = config["datasets"]
    elif files:
        sources = [{"path": path} for path in files]
        directory = Path.cwd()
    else:
        raise ValueError("Preflight requires prompt config or prompt files")
    result = []
    for source in sources:
        data_path = Path(source["path"].split("@[", 1)[0])
        if not data_path.is_absolute():
            data_path = directory / data_path
        with data_path.open() as stream:
            row = next((json.loads(line) for line in stream if line.strip()), None)
        if row is None or row.get(source.get("input_key", "prompt")) is None:
            raise ValueError(f"Missing prompt data in {data_path}")
        if row.get(source.get("label_key", "label")) is None:
            raise ValueError(f"Missing converted RL label in {data_path}")
        if (row.get(source.get("metadata_key", "metadata")) or {}).get("eval_only"):
            raise ValueError(f"Evaluation-only data cannot be used for training: {data_path}")
        result.append((source, row))
    return result


def check_reward_runtime(sources: list[tuple[dict, dict]]) -> None:
    # Reward dependencies are intentionally loaded inside the training image.
    from miles.utils.function_registry import load_function

    code_checked = False
    for source, row in sources:
        hook = source.get("custom_rm_path")
        if hook is None:
            continue
        load_function(hook)
        metadata = row.get(source.get("metadata_key", "metadata")) or {}
        if hook == "experiments.src.nemo_gym.bridge.reward.reward" and metadata.get("verifier") not in {
            "nemo_gym_tau2",
            "nemo_gym_workplace",
        }:
            from experiments.src.nemo_gym.bridge.payloads import verify_payload

            sample = SimpleNamespace(
                prompt=row.get(source.get("input_key", "prompt")),
                response="probe",
                label=row.get(source.get("label_key", "label")),
                metadata=metadata,
            )
            service, _ = verify_payload(sample)
            check_nemo_service(service)
            if service == "code_gen" and not code_checked:
                asyncio.run(check_code_probe())
                code_checked = True


def check_code_probe():
    from experiments.src.nemo_gym.bridge.reward import reward

    async def probe():
        samples = [
            SimpleNamespace(
                prompt="Increment the input.",
                response=response,
                metadata={"verifier": "python_code", "unit_tests": {"inputs": ["2"], "outputs": ["3"]}},
            )
            for response in ("```python\nprint(int(input())+1)\n```", "```python\nprint(0)\n```")
        ]
        if await reward(None, samples) != [1.0, 0.0]:
            raise RuntimeError("NeMo Gym Code failed its known-answer probe")

    return probe()


def check_nemo_service(service: str) -> None:
    import urllib.request

    from experiments.src.nemo_gym.bridge.client import endpoints

    manifest = endpoints()
    if manifest.get("verifier_errors") != "raise":
        raise ValueError("NeMo Gym runtime must propagate verifier errors, not return reward 0")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(manifest["services"][service] + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"NeMo Gym service is not ready: {service}")


def check_tool_runtime(sources: list[tuple[dict, dict]]) -> None:
    from experiments.src.nemo_gym.bridge.payloads import AGENT_VERIFIERS

    for source, row in sources:
        metadata = row.get(source.get("metadata_key", "metadata")) or {}
        verifier = metadata.get("verifier")
        if verifier in {"areal_tau2_environment", "tau3_environment", "workplace_environment"}:
            raise ValueError("Reconvert legacy tool data to native NeMo Gym rows")
        if verifier not in AGENT_VERIFIERS:
            continue
        native = metadata["nemo_gym"]
        if not native["request"].get("responses_create_params"):
            raise ValueError("NeMo Gym agent needs the native request, including its policy prompt")
        if source.get("custom_rm_path") != "experiments.src.nemo_gym.bridge.reward.reward":
            raise ValueError("NeMo Gym agent sources require the NeMo Gym terminal reward hook")
        check_nemo_service(native["service"])


def check_user_simulator() -> None:
    import urllib.request

    from experiments.src.nemo_gym.runtime.serve import model_config

    endpoint = model_config(os.environ["TAU_USER_MODEL_CONFIG"])
    request = urllib.request.Request(
        endpoint["openai_base_url"] + "/models", headers={"Authorization": "Bearer " + endpoint["openai_api_key"]}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=30) as response:
        data = json.load(response)
    if endpoint["openai_model"] not in {item["id"] for item in data["data"]}:
        raise RuntimeError("Local user simulator did not serve its published model identity")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check multi-env data headers and reward runtimes before Ray starts")
    parser.add_argument("--config", required=True, help="JSON/YAML path or base64: inline JSON from the job script")
    parser.add_argument("--require-user-simulator", action="store_true")
    args = parser.parse_args()
    sources = read_sources(args.config, files=json.loads(os.environ.get("PROMPT_DATA_FILES") or "null"))
    check_reward_runtime(sources)
    check_tool_runtime(sources)
    if args.require_user_simulator:
        check_user_simulator()
    if os.environ.get("SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS") == "1":
        from sglang.srt.server_args import ServerArgs

        if not hasattr(ServerArgs, "enable_response_weight_version_segments"):
            raise RuntimeError("Select a SGLang provenance-patched container for first-prefill logging")
    print("Multi-env data headers, reward runtime, and provenance arguments: OK", flush=True)


if __name__ == "__main__":
    main()
