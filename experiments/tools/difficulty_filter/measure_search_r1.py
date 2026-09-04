#!/usr/bin/env python3
"""Measure Search-R1 pass rates against an already-running SGLang server.

The expensive policy and retriever servers are owned by
``run_measure_search_r1.sbatch``. This process only loads the prompt file, renders
the model's chat template, and calls the same ``generate_with_search.generate``
and exact-match reward functions used during training.

Output is one resumable JSON record per prompt.  Full trajectories are kept in
an optional audit file; the primary file contains only scores and operational
statistics, so a resumable full-dataset difficulty measurement stays compact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


ACTION_STOPS = ("</search>", "</answer>")
PROTOCOL_TAGS = (
    "<think>",
    "</think>",
    "<search>",
    "</search>",
    "<information>",
    "</information>",
    "<answer>",
    "</answer>",
)


@dataclass(frozen=True)
class EpisodeResult:
    """Serializable result of one sampled Search-R1 trajectory."""

    reward: float
    status: str
    response: str
    generated_tokens: int
    observation_tokens: int
    turns: int
    search_calls: int

    @property
    def answered(self) -> bool:
        return "</answer>" in self.response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-data", required=True, help="Miles-format Search-R1 JSONL or parquet")
    parser.add_argument("--output", required=True, help="resumable per-prompt JSONL")
    parser.add_argument("--model-path", required=True, help="HF checkpoint used by SGLang and the tokenizer")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--label-key", default="reward_model")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="generation cap per LLM turn")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--search-topk", type=int, default=3)
    parser.add_argument("--search-concurrency", type=int, default=4)
    parser.add_argument("--search-timeout", type=int, default=60)
    parser.add_argument("--search-max-attempts", type=int, default=3)
    parser.add_argument("--format-score", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4, help="sample trajectories in flight")
    parser.add_argument("--request-timeout", type=int, default=3600, help="whole-trajectory timeout")
    parser.add_argument("--dp-size", type=int, default=1, help="number of replicas behind the SGLang endpoint")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test prompt cap")
    parser.add_argument("--correct-threshold", type=float, default=0.5)
    parser.add_argument("--dump-responses", default=None, help="optional full-trajectory audit JSONL")
    parser.add_argument("--dump-limit", type=int, default=8)
    parser.add_argument("--policy", default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "n_samples",
        "max_new_tokens",
        "max_turns",
        "search_topk",
        "search_concurrency",
        "search_timeout",
        "search_max_attempts",
        "concurrency",
        "request_timeout",
        "dp_size",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive, got {value}")
    if not 0 <= args.format_score <= 1:
        raise SystemExit(f"--format-score must be between 0 and 1, got {args.format_score}")
    if not 0 <= args.correct_threshold <= 1:
        raise SystemExit(f"--correct-threshold must be between 0 and 1, got {args.correct_threshold}")


def _configure_search_environment(args: argparse.Namespace) -> None:
    values = {
        "SEARCH_R1_SEARCH_URL": args.search_url,
        "SEARCH_R1_MAX_TURNS": args.max_turns,
        "SEARCH_R1_TOPK": args.search_topk,
        "SEARCH_R1_SEARCH_CONCURRENCY": args.search_concurrency,
        "SEARCH_R1_SEARCH_TIMEOUT": args.search_timeout,
        "SEARCH_R1_SEARCH_MAX_ATTEMPTS": args.search_max_attempts,
        "SEARCH_R1_FORMAT_SCORE": args.format_score,
        # Difficulty measurement and offline EM do not consume behavior-policy
        # log-probs. Avoid asking SGLang to score every generated token; training
        # leaves this unset and retains log-probs for TIS/token alignment.
        "SEARCH_R1_RETURN_LOGPROB": 0,
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def _load_runtime_modules():
    repo_root = Path(__file__).resolve().parents[3]
    search_r1_dir = repo_root / "examples" / "experimental" / "search-r1"
    sys.path.insert(0, str(search_r1_dir))

    import generate_with_search  # noqa: PLC0415
    from miles.utils.chat_template_utils import apply_chat_template  # noqa: PLC0415
    from miles.utils.types import Sample  # noqa: PLC0415

    return generate_with_search, apply_chat_template, Sample


def _server_address(server_url: str) -> tuple[str, int]:
    parsed = urlparse(server_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None or parsed.path not in ("", "/"):
        raise SystemExit(f"--server-url must be an HTTP origin with an explicit port, got {server_url!r}")
    return parsed.hostname, parsed.port


def _generate_args(args: argparse.Namespace) -> SimpleNamespace:
    host, port = _server_address(args.server_url)
    return SimpleNamespace(
        hf_checkpoint=args.model_path,
        chat_template_path=None,
        sglang_server_concurrency=args.concurrency,
        rollout_num_gpus=args.dp_size,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=args.temperature,
        rollout_top_p=args.top_p,
        rollout_top_k=args.top_k,
        rollout_max_response_len=args.max_new_tokens,
        rollout_stop=list(ACTION_STOPS),
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        n_samples_per_prompt=args.n_samples,
        sglang_dp_size=args.dp_size,
        partial_rollout=False,
        sglang_router_ip=host,
        sglang_router_port=port,
        sglang_speculative_algorithm=None,
        use_distributed_post=False,
    )


def protocol_tokenization(tokenizer) -> dict[str, dict]:
    """Verify that ordinary-text protocol tags round-trip exactly.

    Atomic special tokens are not required: SGLang stops on strings and the
    environment tokenizes observations as ordinary context.  Exact round-trip
    encoding is the property those two boundaries require.
    """
    all_special_ids = set(tokenizer.all_special_ids)
    result = {}
    for tag in PROTOCOL_TAGS:
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not token_ids or decoded != tag:
            raise RuntimeError(
                f"protocol tag does not round-trip through the tokenizer: {tag!r} -> {token_ids} -> {decoded!r}"
            )
        result[tag] = {
            "token_ids": token_ids,
            "atomic_special_token": len(token_ids) == 1 and token_ids[0] in all_special_ids,
        }
    return result


def _iter_input_rows(path: str):
    suffix = Path(path).suffix
    if suffix == ".jsonl":
        with open(path) as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to read Search-R1 parquet prompt data")
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            yield from batch.to_pylist()
        return
    raise SystemExit(f"unsupported prompt data {path!r}; expected .jsonl or .parquet")


def _load_rows(path: str, *, input_key: str, label_key: str, metadata_key: str, limit: int | None) -> list[dict]:
    rows = []
    for index, raw in enumerate(_iter_input_rows(path)):
        rows.append(
            {
                "index": index,
                "prompt": raw[input_key],
                "label": raw[label_key],
                "metadata": dict(raw.get(metadata_key) or {}),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _done_indices(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as source:
        for line in source:
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def _finalize_output(path: str) -> list[dict]:
    by_index = {}
    with open(path) as source:
        for line in source:
            try:
                row = json.loads(line)
                by_index[int(row["index"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    temporary = f"{path}.tmp"
    with open(temporary, "w") as destination:
        for index in sorted(by_index):
            destination.write(json.dumps(by_index[index], ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return [by_index[index] for index in sorted(by_index)]


def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else math.nan


def build_record(index: int, row: dict, episodes: list[EpisodeResult], correct_threshold: float) -> dict:
    rewards = [episode.reward for episode in episodes]
    n_correct = sum(reward >= correct_threshold for reward in rewards)
    targets = row["label"].get("ground_truth", {}).get("target") if isinstance(row["label"], dict) else None
    return {
        "index": index,
        "source": row["metadata"].get("source"),
        "question": row["metadata"].get("question"),
        "label": targets,
        "pass_rate": n_correct / len(episodes),
        "n_correct": n_correct,
        "n_samples": len(episodes),
        "response_len_mean": _mean([episode.generated_tokens for episode in episodes]),
        "observation_len_mean": _mean([episode.observation_tokens for episode in episodes]),
        "total_response_len_mean": _mean(
            [episode.generated_tokens + episode.observation_tokens for episode in episodes]
        ),
        "truncated_frac": sum(episode.status == "truncated" for episode in episodes) / len(episodes),
        "search_calls_mean": _mean([episode.search_calls for episode in episodes]),
        "turns_mean": _mean([episode.turns for episode in episodes]),
        "searched_frac": sum(episode.search_calls > 0 for episode in episodes) / len(episodes),
        "answered_frac": sum(episode.answered for episode in episodes) / len(episodes),
        "rewards": rewards,
        "statuses": [episode.status for episode in episodes],
    }


def _episode_from_sample(sample, reward: float) -> EpisodeResult:
    loss_mask = sample.loss_mask or []
    generated_tokens = sum(int(value != 0) for value in loss_mask)
    metadata = sample.metadata or {}
    return EpisodeResult(
        reward=float(reward),
        status=sample.status.value,
        response=sample.response,
        generated_tokens=generated_tokens,
        observation_tokens=len(loss_mask) - generated_tokens,
        turns=int(metadata.get("search_r1_turns", 0)),
        search_calls=int(metadata.get("search_r1_search_calls", 0)),
    )


async def _run_episode(runtime, generate_args, sampling_params, row, rendered_prompt, semaphore, timeout):
    search_module, _, sample_type = runtime
    sample = sample_type(
        index=row["index"],
        prompt=rendered_prompt,
        label=row["label"],
        metadata=dict(row["metadata"]),
    )
    async with semaphore:
        sample = await asyncio.wait_for(
            search_module.generate(generate_args, sample, sampling_params.copy()),
            timeout=timeout,
        )
    if sample.status.value == "aborted":
        detail = (sample.metadata or {}).get("search_r1_error", "unknown generation failure")
        raise RuntimeError(f"trajectory aborted: {detail}")
    reward = await search_module.reward_func(generate_args, sample)
    return _episode_from_sample(sample, reward)


async def _measure_one(runtime, args, generate_args, state, row, semaphore, output, audit, lock, counters):
    _, apply_chat_template, _ = runtime
    rendered_prompt = apply_chat_template(
        row["prompt"],
        tokenizer=state.tokenizer,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        episode_results = await asyncio.gather(
            *(
                _run_episode(
                    runtime,
                    generate_args,
                    state.sampling_params,
                    row,
                    rendered_prompt,
                    semaphore,
                    args.request_timeout,
                )
                for _ in range(args.n_samples)
            ),
            return_exceptions=True,
        )
        failures = [result for result in episode_results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(
                f"{len(failures)}/{args.n_samples} trajectories failed; first error: "
                f"{type(failures[0]).__name__}: {failures[0]}"
            )
        episodes = episode_results
        record = build_record(row["index"], row, episodes, args.correct_threshold)
    except Exception as error:  # noqa: BLE001 - preserve other prompts and make the job fail after flushing them
        async with lock:
            counters["failed"] += 1
            counters["finished"] += 1
            print(f"[error] index={row['index']} {type(error).__name__}: {error}", flush=True)
        return

    async with lock:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        if audit is not None and counters["dumped"] < args.dump_limit:
            audit.write(
                json.dumps(
                    {
                        "index": row["index"],
                        "prompt_text": rendered_prompt,
                        "label": row["label"],
                        "episodes": [asdict(episode) for episode in episodes],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            audit.flush()
            counters["dumped"] += 1
        counters["done"] += 1
        counters["finished"] += 1
        stride = max(1, counters["total"] // 10)
        if counters["finished"] % stride == 0 or counters["finished"] == counters["total"]:
            elapsed = time.monotonic() - counters["started"]
            rate = counters["finished"] / max(elapsed, 1e-9)
            remaining = (counters["total"] - counters["finished"]) / max(rate, 1e-9)
            print(
                f"[{counters['finished']}/{counters['total']}] {rate:.2f} prompt/s "
                f"eta={remaining / 60:.1f}m failed={counters['failed']}",
                flush=True,
            )


def _write_meta(args: argparse.Namespace, tokenization: dict) -> None:
    metadata = {
        "policy": args.policy or Path(args.model_path).name,
        "model_path": args.model_path,
        "prompt_data": args.prompt_data,
        "rm_type": "search_r1",
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens_per_turn": args.max_new_tokens,
        "max_turns": args.max_turns,
        "search_topk": args.search_topk,
        "search_url": args.search_url,
        "format_score": args.format_score,
        "correct_threshold": args.correct_threshold,
        "protocol_tokenization": tokenization,
    }
    path = str(Path(args.output).with_suffix(".meta.json"))
    if Path(args.output).exists() and Path(args.output).stat().st_size:
        if not Path(path).exists():
            raise SystemExit(f"refusing to resume {args.output}: missing provenance sidecar {path}")
        with open(path) as source:
            previous = json.load(source)
        if previous != metadata:
            changed = sorted(key for key in set(previous) | set(metadata) if previous.get(key) != metadata.get(key))
            raise SystemExit(
                f"refusing to mix incompatible Search-R1 measurements in {args.output}; "
                f"changed metadata keys: {changed}"
            )
    with open(path, "w") as destination:
        json.dump(metadata, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    print(f"meta -> {path}", flush=True)


async def _measure_rows(runtime, args, generate_args, state, rows, semaphore, output, audit, lock, counters):
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)

    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _measure_one(
                    runtime,
                    args,
                    generate_args,
                    state,
                    row,
                    semaphore,
                    output,
                    audit,
                    lock,
                    counters,
                )
            finally:
                queue.task_done()

    # The training parquet has ~170k prompts. Creating every prompt/sample task
    # up front would allocate more than a million coroutines at n=8. Keep only a
    # bounded prompt window while the trajectory semaphore controls server load.
    worker_count = min(len(rows), args.concurrency)
    await asyncio.gather(*(worker() for _ in range(worker_count)))


async def main_async(args: argparse.Namespace) -> None:
    _validate_args(args)
    _configure_search_environment(args)
    runtime = _load_runtime_modules()
    search_module, _, _ = runtime
    generate_args = _generate_args(args)
    # RolloutManager normally owns this lifecycle.  Offline evaluation calls the
    # rollout function directly, so it must create the shared httpx client before
    # generate_with_search reaches miles.utils.http_utils.post.
    from miles.utils.http_utils import init_http_client  # noqa: PLC0415

    init_http_client(generate_args)
    state = search_module.GenerateState(generate_args)

    tokenization = protocol_tokenization(state.tokenizer)
    print(f"protocol tokenization: {json.dumps(tokenization, sort_keys=True)}", flush=True)

    rows = _load_rows(
        args.prompt_data,
        input_key=args.input_key,
        label_key=args.label_key,
        metadata_key=args.metadata_key,
        limit=args.limit,
    )
    if not rows:
        raise SystemExit(f"no prompts found in {args.prompt_data}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_meta(args, tokenization)
    done = _done_indices(args.output)
    todo = [row for row in rows if row["index"] not in done]
    print(f"prompts={len(rows)} already_measured={len(done)} todo={len(todo)}", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counters = {
        "done": 0,
        "failed": 0,
        "dumped": 0,
        "finished": 0,
        "total": len(todo),
        "started": time.monotonic(),
    }
    audit_mode = "a" if done else "w"
    audit = open(args.dump_responses, audit_mode) if args.dump_responses and todo else None
    try:
        with open(args.output, "a") as output:
            await _measure_rows(
                runtime,
                args,
                generate_args,
                state,
                todo,
                semaphore,
                output,
                audit,
                lock,
                counters,
            )
    finally:
        if audit is not None:
            audit.close()

    records = _finalize_output(args.output)
    print(f"records={len(records)} newly_measured={counters['done']} failures={counters['failed']}", flush=True)
    if counters["failed"]:
        raise SystemExit(f"{counters['failed']} prompts failed; output is resumable, rerun this benchmark")
    if len(records) != len(rows):
        raise SystemExit(f"incomplete output: expected {len(rows)} records, found {len(records)}")


def main() -> int:
    asyncio.run(main_async(parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
