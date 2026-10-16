import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from experiments.src.datasets.tau2.prepare import adapt_tau2, official_splits
from experiments.src.nemo_gym.bridge.agent import run
from experiments.src.nemo_gym.bridge.reward import reward
from experiments.src.nemo_gym.runtime.serve import check_install


def free_ports():
    for base in range(23100, 25000, 10):
        sockets = []
        try:
            for port in range(base, base + 6):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("No local port block for the runtime probe")


def check_assets(root):
    manifest = check_install(root)
    if not manifest["assets_sha256"]:
        raise RuntimeError("No recorded NeMo Gym assets")
    for name, expected in manifest["assets_sha256"].items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"NeMo Gym asset checksum mismatch: {name}")
    splits = official_splits()
    return {domain: {split: len(ids) for split, ids in values.items()} for domain, values in splits.items()}


async def verify_rewards():
    results = {}
    cases = [
        ("mcqa", "A", {"verifier": "mcqa_regex"}, "<think>Answer: B</think>Answer: A", "Answer: B"),
        (
            "if",
            "",
            {
                "verifier": "ifeval_g",
                "prompt_text": "Say orchid",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["orchid"]}],
            },
            "<think>lily</think>orchid",
            "<think>orchid</think>lily",
        ),
        (
            "code",
            "",
            {"verifier": "python_code", "unit_tests": {"inputs": ["2\n"], "outputs": ["4\n"]}},
            "```python\nprint(int(input()) * 2)\n```",
            "```python\nprint(0)\n```",
        ),
    ]
    for name, label, metadata, correct, wrong in cases:
        values = []
        for response in (correct, wrong):
            sample = SimpleNamespace(prompt="probe", label=label, metadata=dict(metadata), response=response)
            values.append(await reward(None, sample))
        if values != [1.0, 0.0]:
            raise RuntimeError(f"{name} correct/incorrect probe failed: {values}")
        results[name] = values
    sample = SimpleNamespace(
        prompt="probe", label="", metadata=cases[-1][2].copy(), response="```python\nwhile True: pass\n```"
    )
    value = await reward(None, sample)
    failure = sample.metadata["nemo_gym_result"]
    if value is not None or failure.get("failure_kind") != "execution_time_limit":
        raise RuntimeError(f"Code timeout was not kept unscored: {failure}")
    results["code_timeout"] = failure["failure_kind"]
    return results


async def verify_runtime(root, state):
    calls = {"user": 0, "policy": 0}

    async def model(request):
        body = await request.json()
        kind = "policy" if "/sessions/" in request.path else "user"
        calls[kind] += 1
        text = (
            "How can I help you?"
            if kind == "policy"
            else ("I need help with my order." if calls[kind] == 1 else "###STOP###")
        )
        return web.json_response(
            {
                "id": "probe",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "probe"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        )

    async def models(request):
        return web.json_response({"data": [{"id": "probe"}]})

    app = web.Application()
    app.router.add_get("/sessions/probe/v1/models", models)
    app.router.add_post("/sessions/probe/v1/chat/completions", model)
    app.router.add_post("/v1/chat/completions", model)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    user_config = state / "user.json"
    user_config.write_text(json.dumps({"openai_base_url": f"http://127.0.0.1:{port}/v1", "openai_model": "probe"}))
    process = None
    try:
        with (state / "runtime.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.src.nemo_gym.runtime.serve",
                    "--root",
                    str(root),
                    "--state-dir",
                    str(state),
                    "--host",
                    "127.0.0.1",
                    "--port-base",
                    str(free_ports()),
                    "--user-model-config",
                    str(user_config),
                    "--max-steps",
                    "6",
                    "--code-timeout",
                    "2",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            for _ in range(330):
                if process.poll() is not None:
                    raise RuntimeError(f"NeMo Gym exited; inspect {state}")
                if (state / "endpoints.json").exists():
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError(f"NeMo Gym readiness timed out; inspect {state}")
            os.environ["NEMO_GYM_ENDPOINTS"] = str(state / "endpoints.json")
            os.environ["NEMO_GYM_REQUEST_TIMEOUT"] = "60"
            results = await verify_rewards()
            row = json.loads((root / "benchmarks/tau2/data/tau2_benchmark.jsonl").read_text().splitlines()[0])
            row["config"]["max_steps"] = 6
            outcome = await run(
                f"http://127.0.0.1:{port}/sessions/probe",
                None,
                {"temperature": 0},
                adapt_tau2(row, split="test")["metadata"],
            )
            if outcome["nemo_gym_result"]["status"] != "scored" or calls["user"] < 2 or calls["policy"] < 1:
                raise RuntimeError(f"Tau2 multi-turn probe failed: {outcome}, {calls}")
            results["tau2"] = {"calls": calls, "reward": outcome["nemo_gym_result"]["reward"]}
            return results
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            await asyncio.to_thread(process.wait, 60)
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Verify installed assets and native CPU grading before SIF publication"
    )
    parser.add_argument("--root", type=Path, default=Path("/opt/nemo-gym"))
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    state = args.state_dir or Path(tempfile.mkdtemp(prefix="nemo-gym-verify-"))
    state.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        NEMO_GYM_OFFLINE="1",
        NLTK_DATA=str(args.root / "nltk_data"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    results = {"tau_splits": check_assets(args.root), **asyncio.run(verify_runtime(args.root, state))}
    (state / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"state_dir": str(state), **results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
