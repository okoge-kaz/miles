#!/bin/bash
# Evaluate one policy endpoint on SWE-bench Verified through the same
# NeMo-Gym harness training uses, so baseline and post-training numbers are
# produced by an identical grading path and are directly comparable.
#
# Usage:
#   ./run_eval.sh baseline        http://<sglang-host>:<port>/v1
#   ./run_eval.sh step0040        http://<sglang-host>:<port>/v1
#   EVAL_SET=full ./run_eval.sh final http://<sglang-host>:<port>/v1
#
# The endpoint is any OpenAI-compatible server: SGLang serving the checkpoint
# under test, or inference-api.nvidia.com/v1 for an external-API reference
# point.

set -euo pipefail

LABEL="${1:?usage: run_eval.sh <label> <policy-base-url>}"
POLICY_BASE_URL="${2:?usage: run_eval.sh <label> <policy-base-url>}"

EXP_DIR="${EXP_DIR:-$PWD}"
MILES_DIR="${MILES_DIR:-$(git -C "$EXP_DIR" rev-parse --show-toplevel)}"
EVAL_SET="${EVAL_SET:-dev}"                       # dev (100) or full (500)
CONCURRENCY="${CONCURRENCY:-16}"
NEMO_GYM_URL="${NEMO_GYM_URL:-$(cat "$EXP_DIR/.nemo_gym_url")}"

INPUT="$EXP_DIR/data/eval_verified_${EVAL_SET}.jsonl"
OUT_DIR="$EXP_DIR/results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/verified_${EVAL_SET}_${LABEL}.jsonl"

echo "eval  set=${EVAL_SET}  label=${LABEL}"
echo "  gym    ${NEMO_GYM_URL}"
echo "  policy ${POLICY_BASE_URL}"
echo "  out    ${OUT}"

# temperature/top_p match the training rollout sampling, so eval measures the
# same distribution GRPO is optimizing rather than a greedy variant of it.
NEMO_GYM_URL="$NEMO_GYM_URL" python "$MILES_DIR/examples/experimental/nemo-gym/eval_nemogym_via_api.py" \
    --input "$INPUT" \
    --policy-base-url "$POLICY_BASE_URL" \
    --concurrency "$CONCURRENCY" \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 16384 \
    --output "$OUT"

python - "$OUT" <<'PY'
import json, math, sys

rows = [json.loads(line) for line in open(sys.argv[1])]
n = len(rows)
solved = sum(1 for r in rows if r["reward"] == 1.0)
errored = sum(1 for r in rows if r["error"])
p = solved / n if n else 0.0
# Normal-approximation 95% interval: with n=100 the half-width is ~+/-9 points,
# which is the whole point of printing it -- do not call a 3-point move a win.
half = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
print(f"\npass@1 {p:.3f} +/- {half:.3f}  ({solved}/{n} solved, {errored} harness errors)")
if errored:
    print("harness errors are NOT model failures - investigate before comparing runs")
PY
