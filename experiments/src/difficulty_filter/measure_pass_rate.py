"""Measure each prompt's pass rate under one policy, writing one JSON line per prompt.

Runs against an already-serving SGLang endpoint (see run_measure.sbatch, which
starts one and waits for it). Generation and scoring are deliberately split from
the filtering itself: the measurement is the expensive, policy-specific half and
is worth caching, while re-selecting a different window from a finished
measurement is `apply_filter.py` and costs nothing.

Scoring goes through `miles.rollout.rm_hub.batched_async_rm`, not a local
reimplementation, so a prompt is graded here by exactly the verifier that will
grade it during training -- including the `boxed_` prefix handling and the
`</think>` split inside the deepscaler reward. A filter calibrated with a
different grader than the trainer uses would be worse than no filter.

Resumable: results are appended and flushed per prompt, and a restart skips
indices already present in the output. A 4 h partition limit is therefore not a
correctness concern, only a wall-clock one.
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from types import SimpleNamespace

import aiohttp

# The container mounts this checkout at /root/miles; make `experiments.*` and
# `miles.*` importable when the driver is run as a plain script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from miles.rollout.rm_hub import batched_async_rm  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

from experiments.src.difficulty_filter.pass_rate import (  # noqa: E402
    DEFAULT_CORRECT_THRESHOLD,
    DEFAULT_PASS_RATE_MAX,
    DEFAULT_PASS_RATE_MIN,
    PassRateRecord,
    pass_rate_from_rewards,
    summarize,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt-data", required=True, help="input jsonl (one prompt per line)")
    p.add_argument("--output", required=True, help="output jsonl of PassRateRecord, appended and resumable")
    p.add_argument("--model-path", required=True, help="HF dir, used for the tokenizer's chat template")
    p.add_argument("--server-url", default="http://127.0.0.1:30000")
    p.add_argument("--input-key", default="prompt")
    p.add_argument("--label-key", default="label")
    # Same name and default as the training flag, so a sweep and its recipe
    # cannot disagree about where the per-row tools live.
    p.add_argument("--tool-key", default="tools")
    # Must match the training recipe's --rm-type. `deepscaler` and `math` grade
    # the boxed answer identically; `deepscaler` additionally *requires* a
    # `</think>` (or `###Response`) delimiter and returns 0 without one
    # (rm_hub/deepscaler.py:36-44), so it silently zeroes every response from a
    # non-thinking model. `verifier_preflight` below refuses to start on that
    # mismatch rather than producing an all-zero pass-rate file.
    p.add_argument("--rm-type", default="math", help="must match the training recipe's --rm-type")
    p.add_argument("--custom-rm-path", default=None)
    p.add_argument("--reward-key", default=None, help="for rm types returning a dict, e.g. 'score' for --rm-type dapo")
    # Defaults mirror experiments/math_sync: n_samples_per_prompt 8, temperature 1.
    # The measured pass rate is then the same statistic the trainer will see per
    # group, so the window maps onto training batches without rescaling.
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    # Must equal the training recipe's --rollout-max-response-len. A truncated
    # sample scores 0 under every rule-based verifier, so measuring with a
    # smaller budget does not add noise -- it adds a *directional* bias that
    # mislabels long-solution problems as too hard and drops exactly the
    # prompts a math curriculum most wants to keep.
    p.add_argument("--max-new-tokens", type=int, default=24576)
    p.add_argument("--concurrency", type=int, default=64, help="prompts in flight; each expands to --n-samples")
    p.add_argument("--limit", type=int, default=None, help="stop after this many prompts (smoke runs)")
    p.add_argument("--correct-threshold", type=float, default=DEFAULT_CORRECT_THRESHOLD)
    p.add_argument("--pass-rate-min", type=float, default=DEFAULT_PASS_RATE_MIN, help="reporting only")
    p.add_argument("--pass-rate-max", type=float, default=DEFAULT_PASS_RATE_MAX, help="reporting only")
    p.add_argument("--request-timeout", type=int, default=3600)
    # A pass-rate file is only as trustworthy as the grading behind it, and the
    # grading is invisible in the aggregate. Dumping the raw responses next to
    # the rewards for a handful of prompts is what makes "0 correct" separable
    # into "the model is wrong" and "the verifier is not seeing the answer".
    p.add_argument("--dump-responses", default=None, help="write full responses + rewards here for auditing")
    p.add_argument("--dump-limit", type=int, default=8, help="number of prompts to dump")
    p.add_argument("--policy", default=None, help="name this measurement is filed under (default: model dir name)")
    # Some verifiers admit no synthesizable correct answer. `ifbench` grades
    # arbitrary per-row constraints ("4 paragraphs, the 3rd starting with
    # 'crash'"), so no generic probe can satisfy them and a strict preflight
    # would refuse every such sweep. `warn` keeps the diagnostic without the
    # veto; `off` skips the probes entirely.
    p.add_argument("--preflight", choices=("strict", "warn", "off"), default="strict")
    return p.parse_args()


def finalize_output(path):
    """Rewrite the append log as a sorted, de-duplicated file.

    During the sweep, lines land in completion order and a resumed job appends
    after a hole, so the raw log is neither sorted nor diffable. Sorting once at
    the end makes the artifact deterministic: the same prompts measured with the
    same parameters produce a byte-identical file regardless of how many times
    the job was interrupted. Rewrites via a temp file so an interrupted finalize
    cannot destroy the measurements.
    """
    by_index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_index[row["index"]] = row

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for index in sorted(by_index):
            f.write(json.dumps(by_index[index]) + "\n")
    os.replace(tmp, path)
    return [PassRateRecord(**row) for _, row in sorted(by_index.items())]


def meta_path(output_path):
    return os.path.splitext(output_path)[0] + ".meta.json"


def write_meta(args):
    """Record what produced this measurement, next to it.

    A pass rate is meaningless without the policy and sampling parameters behind
    it, and those are exactly what gets lost when a file is copied or revisited
    weeks later. `apply_filter` reads this back so the annotation it writes into
    the dataset carries the same provenance.
    """
    meta = {
        "policy": args.policy or os.path.basename(os.path.normpath(args.model_path)),
        "model_path": args.model_path,
        "prompt_data": args.prompt_data,
        "rm_type": args.rm_type,
        "custom_rm_path": args.custom_rm_path,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "correct_threshold": args.correct_threshold,
    }
    with open(meta_path(args.output), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {meta_path(args.output)}: {json.dumps(meta)}", flush=True)
    return meta


def load_prompts(path, input_key, label_key, limit=None, tool_key="tools"):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # metadata rides along: gpqa reads valid_letters from it, ifbench reads
            # instruction_id_list/kwargs. rm_hub.async_rm forwards sample.metadata, so a
            # verifier needing more than a bare label gets it without a driver change.
            metadata = dict(row.get("metadata") or {})
            # A prompt file written for miles keeps its tools at the top level,
            # because --tool-key reads them from there (data.py:211-217); the
            # Nemotron converters happen to tuck them into metadata instead. Take
            # either, or the tools silently vanish from the rendered prompt and
            # every row scores 0 for want of a function to call.
            if tool_key and tool_key in row and "tools" not in metadata:
                metadata["tools"] = row[tool_key]
            rows.append(
                {
                    "index": i,
                    "prompt": row[input_key],
                    "label": row.get(label_key),
                    "metadata": metadata,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_done_indices(path):
    """Indices already measured. Malformed trailing lines (a job killed mid-write)
    are dropped rather than aborting the resume."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_prompt_text(tokenizer, prompt, tools=None):
    """Mirror miles' own prompt construction (generate_endpoint_utils.py:29-34):
    a chat-formatted prompt goes through the template, a plain string does not.

    `tools` is per-row here rather than global. The agentic tool-use components
    span 838 domains and ship their own tool signatures on every line, so a
    single `--generate-tool-specs-path` (what miles' multi_turn generate function
    expects) cannot describe them; without the row's own tools in the prompt the
    policy has no function to call and would be scored against an expert call it
    was never shown.
    """
    if isinstance(prompt, str):
        return prompt
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(prompt, **kwargs)


def scalar_reward(reward, reward_key=None) -> float:
    """Collapse a reward to a float. `--rm-type dapo` returns
    {"score", "acc", "pred"}, so a dict is normal rather than an error."""
    if isinstance(reward, dict):
        if reward_key is None:
            raise SystemExit(
                f"rm returned a dict {sorted(reward)}; pass --reward-key to say which field is the score"
            )
        return float(reward[reward_key])
    return float(reward)


async def score_responses(args, prompt, label, responses, metadata=None):
    rm_args = SimpleNamespace(rm_type=args.rm_type, custom_rm_path=args.custom_rm_path)
    samples = [Sample(prompt=prompt, response=r, label=label, metadata=dict(metadata or {})) for r in responses]
    rewards = await batched_async_rm(rm_args, samples)
    return [scalar_reward(r, args.reward_key) for r in rewards]


def _build_probes(label, metadata):
    """Synthetic (correct, wrong) responses in the answer formats this label admits.

    The correct form depends on the task, not on the verifier: a boxed value for
    maths, a bare option letter for multiple choice. Probing both and requiring
    that *some* correct form is graded keeps the check useful across rm types
    without hard-coding which one the caller picked.
    """
    correct = {
        "plain_boxed": f"So the answer is.\n\nAnswer: \\boxed{{{label}}}",
        "thinking_boxed": f"<think>work</think>\nAnswer: \\boxed{{{label}}}",
        "bare_answer": f"Answer: {label}",
    }

    # A wrong answer has to be wrong in the same alphabet as the right one, or
    # the verifier may fail to parse it and score 0 for the wrong reason.
    letters = [str(x).upper() for x in (metadata or {}).get("valid_letters") or []]
    if len(label) == 1 and label.upper() in letters:
        other = next((x for x in letters if x != label.upper()), None)
        wrong = f"Answer: {other}" if other else "Answer: \\boxed{-987654321}"
    else:
        wrong = "Answer: \\boxed{-987654321}"
    return correct, wrong


def _custom_preflight_probes(args, label, metadata):
    """Ask a custom verifier's own module for its (correct, wrong) probe pair.

    Convention: a module exporting `--custom-rm-path` may also export
    `build_preflight_probes(label, metadata) -> (correct, wrong)`. Both halves
    have to come from there. The driver cannot invent a correct answer for an
    arbitrary verifier -- "the expert's tool call" is not guessable from a label
    -- and it cannot invent a wrong one either: for a tool-use row whose expert
    replied in text, the generic wrong answer calls no tool and so scores as
    correct, tripping the guard on a false positive.
    """
    if not args.custom_rm_path:
        return None
    module_path = args.custom_rm_path.rsplit(".", 1)[0]
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    builder = getattr(module, "build_preflight_probes", None)
    if builder is None:
        return None
    try:
        return builder(label, metadata)
    except Exception as exc:  # noqa: BLE001 - a broken probe must not block a sweep
        print(f"  preflight probe builder failed: {type(exc).__name__}: {exc}", flush=True)
        return None


async def verifier_preflight(args, label, metadata=None):
    """Refuse to run a sweep whose verifier cannot recognise a correct answer.

    An all-zero pass-rate file is indistinguishable from "this policy is bad at
    this dataset", so a verifier/model mismatch would be discovered only after
    the whole sweep, or worse, after a training run that silently learns
    nothing. The `deepscaler` case is the concrete one this exists for: it gates
    on a `</think>` delimiter that no non-thinking model emits.
    """
    if args.preflight == "off":
        print("verifier preflight: skipped (--preflight off)", flush=True)
        return
    label = label if isinstance(label, str) and label else "37"
    correct, wrong = _build_probes(label, metadata)
    custom_probes = _custom_preflight_probes(args, label, metadata)
    if custom_probes is not None:
        correct = {"verifier_supplied": custom_probes[0]}
        wrong = custom_probes[1]
    elif args.custom_rm_path:
        print(
            f"  no build_preflight_response in {args.custom_rm_path.rsplit('.', 1)[0]}; "
            f"probing with generic answer formats, which a custom verifier may legitimately reject",
            flush=True,
        )
    names = list(correct) + ["clearly_wrong"]
    responses = list(correct.values()) + [wrong]
    scores = dict(zip(names, await score_responses(args, "", label, responses, metadata), strict=True))
    print(f"verifier preflight (--rm-type {args.rm_type}, label={label!r}): {scores}", flush=True)

    threshold = args.correct_threshold
    lenient = args.preflight == "warn"

    def fail(message):
        if lenient:
            print(f"  PREFLIGHT WARNING (--preflight warn): {message}", flush=True)
            return
        raise SystemExit(message)

    if scores["clearly_wrong"] >= threshold:
        fail(f"--rm-type {args.rm_type} scores a clearly wrong answer as correct; refusing to measure")
        return

    graded = [name for name in correct if scores[name] >= threshold]
    if graded:
        print(f"  correct answers are graded in these forms: {graded}", flush=True)
        return

    if scores.get("thinking_boxed", 0.0) >= threshold:
        fail(
            f"--rm-type {args.rm_type} only grades responses containing a '</think>' delimiter, but the policy "
            f"being measured does not emit one, so every response would score 0.\n"
            f"Use --rm-type math (same boxed-answer grading, no thinking-format gate) and make the training "
            f"recipe match."
        )
        return
    fail(
        f"--rm-type {args.rm_type} scored every correct-answer form as 0 ({scores}); "
        f"the verifier and this dataset's answer format disagree."
    )


async def generate_group(session, url, text, sampling_params, timeout):
    payload = {"text": text, "sampling_params": sampling_params}
    async with session.post(f"{url}/generate", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        out = await resp.json()
    # n > 1 returns a list; n == 1 returns a single object.
    return out if isinstance(out, list) else [out]


async def measure_one(session, args, tokenizer, row, sampling_params, sem, out_file, lock, counters):
    async with sem:
        text = build_prompt_text(tokenizer, row["prompt"], (row.get("metadata") or {}).get("tools"))
        try:
            outputs = await generate_group(session, args.server_url, text, sampling_params, args.request_timeout)
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill a 17k-prompt sweep
            async with lock:
                counters["failed"] += 1
                print(f"[warn] index={row['index']} generation failed: {type(exc).__name__}: {exc}", flush=True)
            return

        responses = [o["text"] for o in outputs]
        metas = [o.get("meta_info", {}) for o in outputs]
        completion_tokens = [m.get("completion_tokens", 0) for m in metas]
        truncated = [m.get("finish_reason", {}).get("type") == "length"
                     if isinstance(m.get("finish_reason"), dict)
                     else m.get("finish_reason") == "length"
                     for m in metas]

        rewards = await score_responses(args, row["prompt"], row["label"], responses, row.get("metadata"))

        record = PassRateRecord(
            index=row["index"],
            pass_rate=pass_rate_from_rewards(rewards, args.correct_threshold),
            n_correct=sum(1 for r in rewards if r >= args.correct_threshold),
            n_samples=len(rewards),
            response_len_mean=sum(completion_tokens) / max(1, len(completion_tokens)),
            truncated_frac=sum(truncated) / max(1, len(truncated)),
            label=row["label"] if isinstance(row["label"], str) else None,
            rewards=rewards,
        )

        async with lock:
            out_file.write(json.dumps(record.__dict__) + "\n")
            out_file.flush()
            dump_file = counters.get("dump_file")
            if dump_file is not None and counters["dumped"] < args.dump_limit:
                dump_file.write(
                    json.dumps(
                        {
                            "index": row["index"],
                            "label": row["label"],
                            "prompt_text": text,
                            "responses": responses,
                            "rewards": rewards,
                            "finish_reasons": [m.get("finish_reason") for m in metas],
                            "completion_tokens": completion_tokens,
                        }
                    )
                    + "\n"
                )
                dump_file.flush()
                counters["dumped"] += 1
            counters["done"] += 1
            if counters["done"] % 200 == 0:
                elapsed = time.time() - counters["t0"]
                rate = counters["done"] / max(1e-9, elapsed)
                remaining = (counters["total"] - counters["done"]) / max(1e-9, rate)
                print(
                    f"[{counters['done']}/{counters['total']}] "
                    f"{rate:.2f} prompt/s  eta {remaining / 60:.1f} min  failed={counters['failed']}",
                    flush=True,
                )


async def main_async(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    rows = load_prompts(args.prompt_data, args.input_key, args.label_key, args.limit, args.tool_key)
    await verifier_preflight(args, rows[0]["label"] if rows else None, rows[0].get("metadata") if rows else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_meta(args)

    done = load_done_indices(args.output)
    todo = [r for r in rows if r["index"] not in done]
    print(f"prompts={len(rows)} already_measured={len(done)} todo={len(todo)}", flush=True)

    sampling_params = {
        "n": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    counters = {"done": 0, "failed": 0, "dumped": 0, "dump_file": None, "total": len(todo), "t0": time.time()}
    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
        dump_ctx = open(args.dump_responses, "w") if args.dump_responses else None
        try:
            counters["dump_file"] = dump_ctx
            with open(args.output, "a") as out_file:
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    await asyncio.gather(
                        *(
                            measure_one(session, args, tokenizer, row, sampling_params, sem, out_file, lock, counters)
                            for row in todo
                        )
                    )
        finally:
            if dump_ctx is not None:
                dump_ctx.close()
                counters["dump_file"] = None

    records = finalize_output(args.output)
    stats = summarize(records, args.pass_rate_min, args.pass_rate_max)
    print("\n=== pass-rate summary ===", flush=True)
    for k, v in stats.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}", flush=True)
    if counters["failed"]:
        print(f"  {'generation_failures':22s} {counters['failed']}", flush=True)

    hist = [0] * (args.n_samples + 1)
    for r in records:
        hist[min(r.n_correct, args.n_samples)] += 1
    print(f"\n  n_correct histogram (out of {args.n_samples}):", flush=True)
    for k, c in enumerate(hist):
        bar = "#" * int(60 * c / max(1, max(hist)))
        print(f"    {k}/{args.n_samples}  {c:6d}  {bar}", flush=True)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
