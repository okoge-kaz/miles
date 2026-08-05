#!/usr/bin/env python3
"""Static checks on every recipe, run before spending a GPU slot on one.

    experiments/validate.py

Exists because `bash -n` is not enough. Two bugs reached a real 2-node
allocation and burned it: a runtime_env JSON string that was valid shell and
invalid JSON, and a ray head address derived from `hostname -I`, whose first
entry is the same link-local on every node here. Both are cheap to catch here.

Checks per recipe:
  1. shell syntax of run.sbatch and train.sh
  2. RUNTIME_ENV_JSON actually parses, and carries the vars the task needs
  3. the run.sbatch defaults produce a placement common/placement.sh accepts,
     at the node count its own #SBATCH --nodes declares
  4. max_tokens_per_gpu * cp >= rollout_max_context_len
  5. the four-knob invariant holds at the defaults
  6. flag hygiene: --partial-rollout is colocated-only (--fully-async rejects
     it), async carries the staleness bound and the pause mode, MoE carries R3
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
GPUS_PER_NODE = 8

failures: list[str] = []
checked = 0


def fail(recipe: str, msg: str) -> None:
    failures.append(f"{recipe}: {msg}")


def sh(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=REPO_ROOT)


def knob(text: str, name: str) -> int | None:
    m = re.search(rf'{name}:=(\d+)', text)
    return int(m.group(1)) if m else None


def check_recipe(recipe_dir: Path) -> None:
    global checked
    rel = str(recipe_dir.relative_to(EXPERIMENTS))
    run_sbatch = recipe_dir / "run.sbatch"
    train_sh = recipe_dir / "train.sh"
    is_async = rel.startswith("math_async")
    checked += 1

    for path in (run_sbatch, train_sh):
        r = sh(f"bash -n {path}")
        if r.returncode:
            fail(rel, f"{path.name} is not valid shell: {r.stderr.strip()}")
            return

    run_text = run_sbatch.read_text()
    train_text = train_sh.read_text()

    # 2. runtime_env JSON. Read the heredoc body directly rather than through a
    # shell, so this check cannot itself fail on quoting.
    m = re.search(r"RUNTIME_ENV_JSON=\$\(cat <<JSON\n(.*?)\nJSON\n\)", train_text, re.S)
    if not m:
        fail(rel, "no RUNTIME_ENV_JSON heredoc found")
        return
    body = m.group(1).replace("${HAS_NVLINK}", "1")
    try:
        env_vars = json.loads(body)["env_vars"]
    except Exception as exc:  # noqa: BLE001
        fail(rel, f"RUNTIME_ENV_JSON does not parse: {exc}; got {body!r}")
        return
    if "PYTHONPATH" not in env_vars:
        fail(rel, "RUNTIME_ENV_JSON has no PYTHONPATH")
    refactor = env_vars.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR")
    if is_async and refactor != "1":
        fail(rel, "--fully-async needs MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 in runtime_env")
    if not is_async and refactor is not None:
        fail(rel, "colocated must not set MILES_EXPERIMENTAL_ROLLOUT_REFACTOR")

    # 3. placement, at the node count the recipe declares for itself
    nodes = int(re.search(r"#SBATCH --nodes=(\d+)", run_text).group(1))
    defaults = "\n".join(
        line for line in run_text.splitlines() if re.match(r'^: "\$\{[A-Z_0-9]+:=', line)
    )
    r = sh(
        f"set -e; SLURM_JOB_NUM_NODES={nodes}; GPUS_PER_NODE={GPUS_PER_NODE}\n"
        f"{defaults}\n"
        f"source experiments/common/placement.sh"
    )
    if r.returncode:
        fail(rel, f"placement rejected at -N {nodes}: {(r.stderr or r.stdout).strip()}")

    # 4/5. batch shape and the token budget
    tp = knob(run_text, "TENSOR_PARALLEL_SIZE")
    cp = knob(run_text, "CONTEXT_PARALLEL_SIZE")
    mtpg = knob(run_text, "MAX_TOKENS_PER_GPU")
    ctx = int(re.search(r"--rollout-max-context-len (\d+)", train_text).group(1))
    if mtpg * cp < ctx:
        fail(rel, f"max_tokens_per_gpu {mtpg} * cp {cp} = {mtpg * cp} < context {ctx}")

    save_int = knob(run_text, "SAVE_INTERVAL")
    retain = knob(run_text, "SAVE_RETAIN_INTERVAL")
    # Megatron asserts this at startup (training/arguments.py:841).
    if save_int and retain and retain % save_int != 0:
        fail(rel, f"save_retain_interval {retain} is not a multiple of save_interval {save_int}")

    rb = knob(run_text, "ROLLOUT_BATCH_SIZE")
    n = knob(run_text, "N_SAMPLES_PER_PROMPT")
    gbs = knob(run_text, "GLOBAL_BATCH_SIZE")
    steps = knob(run_text, "NUM_STEPS_PER_ROLLOUT")
    if rb * n != gbs * steps:
        fail(rel, f"four-knob invariant: {rb} * {n} != {gbs} * {steps}")

    # 6. flag hygiene
    has_partial = "--partial-rollout" in train_text
    if is_async and has_partial:
        fail(rel, "--fully-async rejects --partial-rollout (arguments.py:54)")
    if not is_async and not has_partial:
        fail(rel, "colocated recipe lost --partial-rollout")
    for flag in ("--max-weight-staleness", "--pause-generation-mode", "--use-tis"):
        if is_async and flag not in train_text:
            fail(rel, f"async recipe is missing {flag}")
        if not is_async and flag in train_text:
            fail(rel, f"colocated recipe should not carry {flag}")
    if "30b-a3b" in rel and "--use-rollout-routing-replay" not in train_text:
        fail(rel, "MoE recipe is missing R3")
    if "instruct-2507" in rel and "RM_TYPE:=math" not in run_text:
        fail(rel, "non-thinking checkpoint must default to RM_TYPE=math")

    print(f"  {rel:<50} -N {nodes}  tp{tp} cp{cp}  ok")


def main() -> int:
    for common in sorted((EXPERIMENTS / "common").glob("*.sh")):
        r = sh(f"bash -n {common}")
        if r.returncode:
            fail("common", f"{common.name}: {r.stderr.strip()}")

    for run_sbatch in sorted(EXPERIMENTS.glob("math_*/*/*/run.sbatch")):
        check_recipe(run_sbatch.parent)

    print()
    if failures:
        print(f"{len(failures)} problem(s) in {checked} recipes:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"{checked} recipes ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
