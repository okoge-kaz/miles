import argparse
import asyncio
import json
import signal
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from experiments.src.difficulty_filtering.core.artifacts import (
    build_manifest,
    ensure_manifest,
    file_lock,
    read_records,
    repair_trailing_record,
    write_json,
)
from experiments.src.difficulty_filtering.core.config import load_config
from experiments.src.difficulty_filtering.core.gym import serve_verifier, verifier_probe
from experiments.src.difficulty_filtering.core.records import build_record, read_prompts, shard_indices, summarize
from experiments.src.difficulty_filtering.core.runtime import serve
from experiments.src.difficulty_filtering.utils.progress import Progress


def build_prompt_text(tokenizer, prompt):
    # Preserve the legacy Miles single-turn contract: strings are already rendered.
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def parse_outputs(outputs, expected):
    if len(outputs) != expected:
        raise ValueError(f"Incomplete SGLang group: expected {expected}, received {len(outputs)}")
    texts, lengths, truncated = [], [], []
    for output in outputs:
        meta = output["meta_info"]
        reason = meta["finish_reason"]
        reason = reason["type"] if isinstance(reason, dict) else reason
        if reason not in {"stop", "length"}:
            raise ValueError(f"Unsuccessful generation: {reason}")
        if not isinstance(output["text"], str) or type(meta["completion_tokens"]) is not int:
            raise ValueError("Invalid SGLang response")
        texts.append(output["text"])
        lengths.append(meta["completion_tokens"])
        truncated.append(reason == "length")
    return texts, lengths, truncated


async def score_responses(config, row, texts, lengths, truncated, diagnostics=None):
    # GPU-image dependencies stay optional for CPU bookkeeping and launch previews.
    from experiments.src.training.truncation import zero_reward
    from experiments.src.nemo_gym.bridge.reward import reward as gym_reward
    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import Sample

    args = SimpleNamespace(rm_type="deepscaler", custom_rm_path=None)
    rewards = []
    for text, length, is_truncated in zip(texts, lengths, truncated, strict=True):
        sample = Sample(
            prompt=row["prompt"],
            label=row["label"],
            response=text,
            response_length=length,
            metadata=deepcopy(row.get("metadata", {})),
            status=Sample.Status.TRUNCATED if is_truncated else Sample.Status.COMPLETED,
        )
        if config.reward_backend == "deepscaler":
            scorer = zero_reward if config.zero_reward_on_truncated else async_rm
            value = await scorer(args, sample)
        elif is_truncated and config.zero_reward_on_truncated:
            value = 0.0
        else:
            value = await gym_reward(args, sample)
        rewards.append(None if value is None else float(value))
        if diagnostics is not None:
            diagnostics.append((sample.metadata or {}).get("nemo_gym_result", {}))
    return rewards


async def verifier_preflight(config):
    row, texts, expected = verifier_probe(config.reward_backend)
    texts.append(texts[0])
    expected.append(0.0 if config.zero_reward_on_truncated else 1.0)
    rewards = await score_responses(
        config,
        row,
        texts,
        [20] * len(texts),
        [False] * (len(texts) - 1) + [True],
    )
    if rewards != expected:
        raise ValueError(f"{config.reward_backend}/truncation preflight failed: {rewards} != {expected}")
    print(f"{config.reward_backend}/truncation preflight: {rewards}", flush=True)


async def generate_request(session, base_url, input_ids, sampling_params, attempts=5):
    import aiohttp

    # Like RL single_turn.generate: one tokenized prompt and one response per request.
    payload = {"input_ids": input_ids, "sampling_params": sampling_params}
    for attempt in range(attempts):
        try:
            async with session.post(base_url + "/generate", json=payload) as response:
                response.raise_for_status()
                output = await response.json()
            parse_outputs([output], 1)
            return output
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as error:
            if attempt + 1 == attempts:
                raise
            print(f"Generation retry {attempt + 1}/{attempts}: {type(error).__name__}", flush=True)
            await asyncio.sleep(min(2**attempt, 30))


async def measure_one(session, base_urls, config, tokenizer, index, row):
    text = build_prompt_text(tokenizer, row["prompt"])
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    budget = min(config.max_new_tokens, config.max_context_length - len(input_ids))
    if budget <= 0:
        # No response budget: record a fully truncated group without an HTTP call.
        texts, lengths, truncated = [""] * config.n_samples, [0] * config.n_samples, [True] * config.n_samples
    else:
        params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_new_tokens": budget,
            # Match inference_rollout_common.compute_sampling_params and the RL defaults.
            "stop": None,
            "stop_token_ids": None,
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }
        outputs = await asyncio.gather(
            *(
                generate_request(
                    session,
                    base_urls[(index * config.n_samples + sample_index) % len(base_urls)],
                    input_ids,
                    params,
                    config.generation_attempts,
                )
                for sample_index in range(config.n_samples)
            )
        )
        texts, lengths, truncated = parse_outputs(outputs, config.n_samples)
    diagnostics = []
    rewards = await score_responses(config, row, texts, lengths, truncated, diagnostics=diagnostics)
    record = build_record(index, row, rewards, lengths, truncated, config, diagnostics)
    audit = {"index": index, "prompt_text": text, "label": row["label"], "responses": texts, **record}
    return record, audit


async def measure_pending(config, rows, indices, done, directory, base_urls, tokenizer):
    # aiohttp is needed only in the GPU measurement environment.
    import aiohttp

    pending = iter(index for index in indices if index not in done)
    audit_indices = set(indices[: config.audit_prompts])
    completed = len(set(indices) & done.keys())
    progress = Progress(config, {index: done[index] for index in indices if index in done}, len(indices), directory)
    connector = aiohttp.TCPConnector(
        force_close=True,
        limit=config.concurrency * config.n_samples,
    )
    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    with (directory / "pass_rates.jsonl").open("a") as journal, (directory / "audit.jsonl").open("a") as audit_file:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

            async def worker():
                nonlocal completed
                for index in pending:
                    record, audit = await measure_one(session, base_urls, config, tokenizer, index, rows[index])
                    # Writes are synchronous on this event loop, so sibling groups cannot interleave them.
                    if index in audit_indices:
                        audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
                        audit_file.flush()
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
                    completed += 1
                    progress.records[index] = record
                    if completed % config.progress_every == 0:
                        progress.emit()

            workers = [asyncio.create_task(worker()) for _ in range(min(config.concurrency, len(indices)))]
            heartbeat = asyncio.create_task(progress.heartbeat())
            progress.emit()
            try:
                await asyncio.gather(*workers)
            finally:
                for task in [*workers, heartbeat]:
                    task.cancel()
                await asyncio.gather(*workers, heartbeat, return_exceptions=True)
                progress.emit()


def measure(config, shard_index, limit=None):
    rows = read_prompts(config.prompt_data)
    assigned = shard_indices(len(rows), config.num_shards, shard_index)
    indices = list(assigned)[:limit]
    directory = Path(config.measurement_dir) / f"part-{shard_index:03d}"
    with file_lock(directory / ".lock"):
        manifest = {
            "measurement": build_manifest(config),
            "shard_index": shard_index,
            "start": assigned.start,
            "stop": assigned.stop,
        }
        ensure_manifest(directory / "manifest.json", manifest)
        repair_trailing_record(directory / "pass_rates.jsonl")
        repair_trailing_record(directory / "audit.jsonl")
        done = read_records(directory / "pass_rates.jsonl", rows, config, assigned)
        print(f"Shard {shard_index}: selected={len(indices)} already_measured={len(done)}", flush=True)
        if set(indices) - done.keys():
            # Tokenizer loading requires the training image; resume-only checks do not.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
            with serve_verifier(config, directory):
                asyncio.run(verifier_preflight(config))
                with serve(config, directory) as base_urls:
                    asyncio.run(measure_pending(config, rows, indices, done, directory, base_urls, tokenizer))
        records = read_records(directory / "pass_rates.jsonl", rows, config, assigned)
        if set(indices) - records.keys():
            raise ValueError("Measurement is incomplete; resume this shard")
        result = {**summarize(list(records.values()), config.n_samples), "complete": set(records) == set(assigned)}
        write_json(directory / "summary.json", result)
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Measure difficulty with SGLang and the training reward")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    measure(load_config(args.config), args.shard_index, args.limit)


if __name__ == "__main__":
    main()
