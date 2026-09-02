#!/usr/bin/env python3
"""Submit a grid of training jobs for one or more recipes.

    experiments/sweep.py --sweep experiments/sweeps/<name>.txt \\
        --recipe math/async/dapo-math-p10-90/qwen3-4b [--recipe ...] \\
        [-- <extra pbs_submit args>]

Prints the grid and exits. Add --submit to actually submit.

Every point gets its own CONFIG_TAG, which is what names the checkpoint
directory and the wandb group (common/run_identity.sh), so two points can never
resume from each other's optimizer state. The tag is
``sweep-<name>-<knob><value>-...`` over the knobs that vary in this grid.

miles asserts rollout_batch * n_samples == global_batch * num_steps at startup,
so a grid that moves the left-hand side has to move the right-hand side with it
or every job dies on the assert. Whichever of GLOBAL_BATCH_SIZE /
NUM_STEPS_PER_ROLLOUT the sweep does not pin is derived per point; pin both and
they are checked instead. Pinning GLOBAL_BATCH_SIZE keeps the optimizer step the
same size while the rollout grows, which is how DAPO scales; pinning
NUM_STEPS_PER_ROLLOUT grows the step with the rollout.

Sweep file format — one knob per line, values separated by whitespace:

    # comment
    LR                     5e-7 1e-6 2e-6
    N_SAMPLES_PER_PROMPT   8 16
    NUM_STEPS_PER_ROLLOUT  1 2

    # a line with a single value is applied to every point without entering
    # the product, and without appearing in the tag
    MAX_RESPONSE_LEN       16384

A manifest of what was submitted lands in
experiments/outputs/sweeps/<name>-<timestamp>.jsonl for joining job ids back to
their configuration later.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
EXPERIMENT_SCRIPTS = EXPERIMENTS / "scripts"
PBS_SUBMIT = EXPERIMENTS / "common" / "pbs.sh"

# Short forms used in CONFIG_TAG, so a tag stays readable as a directory name.
TAG_ABBREV = {
    "LR": "lr",
    "N_SAMPLES_PER_PROMPT": "n",
    "ROLLOUT_BATCH_SIZE": "rb",
    "GLOBAL_BATCH_SIZE": "gb",
    "NUM_STEPS_PER_ROLLOUT": "step",
    "MAX_RESPONSE_LEN": "len",
    "ROLLOUT_MAX_CONTEXT_LEN": "ctx",
    "MAX_WEIGHT_STALENESS": "stale",
    "OVER_SAMPLING_BATCH_SIZE": "over",
    "TENSOR_PARALLEL_SIZE": "tp",
    "CONTEXT_PARALLEL_SIZE": "cp",
    "EXPERT_PARALLEL_SIZE": "ep",
    "MAX_TOKENS_PER_GPU": "mtpg",
    "ROLLOUT_NUM_GPUS_PER_ENGINE": "eng",
    "SGLANG_MEM_FRACTION": "memfrac",
    "ACTOR_NUM_NODES": "anodes",
    "ACTOR_GPUS_PER_NODE": "agpus",
    "ROLLOUT_NUM_GPUS": "rgpus",
    "PAUSE_GENERATION_MODE": "pause",
    "ADVANTAGE_ESTIMATOR": "adv",
    "EPS_CLIP": "clip",
    "EPS_CLIP_HIGH": "cliphigh",
    "EPS_CLIP_C": "dualclip",
    "RATIO_DENOMINATOR": "denom",
    "IS_CORRECTION": "is",
    "TIS_CLIP": "tisclip",
    "TIS_CLIP_LOW": "tiscliplow",
    "MIS_PROFILE": "mis",
    "USE_OPSM": "opsm",
    "OPSM_DELTA": "opsmdelta",
    "ENTROPY_COEF": "ent",
    "TRAIN_SEED": "tseed",
    "ROLLOUT_SEED": "rseed",
}


def resolve_recipe_path(recipe: str) -> Path:
    """Resolve a public recipe name to its canonical directory."""
    if recipe.startswith("search_r1/async/"):
        return EXPERIMENTS / "search_r1" / recipe.removeprefix("search_r1/")
    return EXPERIMENT_SCRIPTS / recipe


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def parse_sweep(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (knobs that vary, values fixed for the whole grid)."""
    varying: dict[str, list[str]] = {}
    fixed: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, *values = line.split()
        if not values:
            sys.exit(f"{path}:{lineno}: {name} has no values")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            sys.exit(f"{path}:{lineno}: {name!r} is not an environment variable name")
        if len(values) == 1:
            fixed[name] = values[0]
        else:
            varying[name] = values
    if not varying:
        sys.exit(f"{path}: no knob has more than one value, so there is no grid")
    return varying, fixed


def tag_for(point: dict[str, str]) -> str:
    parts = []
    for name, value in point.items():
        abbrev = TAG_ABBREV.get(name, name.lower())
        clean = _SAFE.sub("-", value)
        # "lr1e-6" reads fine, "pausein_place" does not.
        sep = "" if clean[:1].isdigit() else "-"
        parts.append(f"{abbrev}{sep}{clean}")
    return "-".join(parts)


def resolve_batch_shape(env: dict[str, str], recipe: Path) -> str | None:
    """Close the four-knob invariant, in whichever direction the grid leaves open.

        rollout_batch * n_samples = global_batch * num_steps

    miles asserts this at startup, so a grid that moves the left-hand side has to
    move the right-hand side with it. Whichever of global_batch / num_steps the
    sweep does not pin is the one derived; if it pins both, they are checked.

    Returns an error string, or None on success (env is updated in place).
    """
    text = (recipe / "run.sbatch").read_text()

    def knob(name: str) -> int:
        if name in env:
            return int(env[name])
        m = re.search(rf"{name}:=(\d+)", text)
        if not m:
            sys.exit(f"{recipe}: cannot read a default for {name}")
        return int(m.group(1))

    total = knob("ROLLOUT_BATCH_SIZE") * knob("N_SAMPLES_PER_PROMPT")
    has_gbs = "GLOBAL_BATCH_SIZE" in env
    has_steps = "NUM_STEPS_PER_ROLLOUT" in env

    if has_gbs and has_steps:
        gbs, steps = int(env["GLOBAL_BATCH_SIZE"]), int(env["NUM_STEPS_PER_ROLLOUT"])
        if gbs * steps != total:
            return f"global_batch {gbs} * num_steps {steps} != rollout_batch * n_samples = {total}"
        return None

    if has_gbs:
        gbs = int(env["GLOBAL_BATCH_SIZE"])
        if total % gbs:
            return f"rollout_batch * n_samples = {total} is not divisible by global_batch {gbs}"
        env["NUM_STEPS_PER_ROLLOUT"] = str(total // gbs)
        return None

    steps = knob("NUM_STEPS_PER_ROLLOUT")
    if total % steps:
        return f"rollout_batch * n_samples = {total} is not divisible by num_steps {steps}"
    env["GLOBAL_BATCH_SIZE"] = str(total // steps)
    # Recorded even when it came from the recipe's own default, so the printed
    # grid and the manifest describe the whole batch shape rather than half of it.
    env["NUM_STEPS_PER_ROLLOUT"] = str(steps)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, type=Path, help="sweep file")
    ap.add_argument(
        "--recipe",
        required=True,
        action="append",
        help="<task>/<mode>/<dataset>/<model>, repeatable to run the same grid on several models",
    )
    ap.add_argument("--submit", action="store_true", help="actually submit; otherwise print the grid and exit")
    ap.add_argument("--max-jobs", type=int, default=32, help="refuse to submit more points than this (default 32)")
    ap.add_argument("pbs_args", nargs="*", help="extra pbs_submit args, after --")
    args = ap.parse_args()

    varying, fixed = parse_sweep(args.sweep)
    sweep_name = args.sweep.stem

    recipes = []
    for rel in args.recipe:
        path = resolve_recipe_path(rel)
        if not (path / "run.sbatch").is_file():
            sys.exit(f"no such recipe: {rel}")
        recipes.append((rel, path))

    names = list(varying)
    grid = [dict(zip(names, combo)) for combo in itertools.product(*(varying[n] for n in names))]
    total = len(grid) * len(recipes)

    print(f"sweep      {sweep_name}  ({args.sweep})")
    print(f"recipes    {len(recipes)}: " + ", ".join(r for r, _ in recipes))
    print("varying    " + ", ".join(f"{n}={'/'.join(v)}" for n, v in varying.items()))
    if fixed:
        print("fixed      " + ", ".join(f"{k}={v}" for k, v in fixed.items()))
    print(f"points     {len(grid)} per recipe, {total} jobs total")
    print()

    if total > args.max_jobs:
        sys.exit(f"{total} jobs exceeds --max-jobs {args.max_jobs}; raise it deliberately or shrink the grid")

    manifest_dir = EXPERIMENTS / "outputs" / "sweeps"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{sweep_name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    submitted = 0
    for rel, path in recipes:
        for point in grid:
            env = dict(fixed)
            env.update(point)
            config_tag = f"sweep-{sweep_name}-{tag_for(point)}"
            env["CONFIG_TAG"] = config_tag

            if error := resolve_batch_shape(env, path):
                print(f"  SKIP {rel} {config_tag}: {error}")
                continue

            export = "ALL," + ",".join(f"{k}={v}" for k, v in env.items())
            recipe_script = path / "run.sbatch"
            cmd = [
                str(PBS_SUBMIT),
                "--parsable",
                "--profile",
                "gpu",
                f"--export={export}",
                *args.pbs_args,
                str(recipe_script.relative_to(REPO_ROOT)),
            ]

            if not args.submit:
                print(f"  {rel}  {config_tag}  gbs={env["GLOBAL_BATCH_SIZE"]} steps={env["NUM_STEPS_PER_ROLLOUT"]}")
                continue

            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAILED {rel} {config_tag}: {result.stderr.strip()}")
                continue
            job_id = result.stdout.strip()
            submitted += 1
            print(f"  {job_id}  {rel}  {config_tag}")
            with manifest.open("a") as f:
                f.write(json.dumps({"job_id": job_id, "recipe": rel, "sweep": sweep_name, "env": env}) + "\n")

    print()
    if args.submit:
        print(f"submitted {submitted}/{total}")
        print(f"manifest  {manifest.relative_to(REPO_ROOT)}")
    else:
        print("dry run — add --submit to launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
