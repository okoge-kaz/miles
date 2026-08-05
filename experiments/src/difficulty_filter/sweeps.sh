#!/bin/bash
#
# Every (prompt file, verifier) pair we measure pass rates for, in one table.
#
#   experiments/src/difficulty_filter/sweeps.sh smoke            # all, 64 prompts each
#   experiments/src/difficulty_filter/sweeps.sh full  arc-agi    # one, whole file
#   experiments/src/difficulty_filter/sweeps.sh list
#
# The pairing is the part worth version-controlling. A wrong verifier does not
# crash -- it returns 0.0 for every row, and a dataset that looks uniformly
# impossible is indistinguishable from one that is genuinely hard. That failure
# has already happened twice here (`--rm-type deepscaler` on a non-thinking
# model, `--rm-type ifbench` on IFEvalG ids), which is why `PREFLIGHT=strict`
# is the default and why these live in a file instead of in shell history.
#
# `smoke` is the verification pass: 64 prompts, responses dumped, small enough to
# land in minutes. Read the mean pass rate before launching `full` -- 0.000 or
# 1.000 means the verifier, not the policy.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
MODEL="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
NB="experiments.src.nemo_blends"

# name | prompt file (in-container) | verifier | extra exports
#
# Verifier column: `rm:<x>` selects a built-in --rm-type, anything else is a
# dotted --custom-rm-path.
read -r -d '' SWEEPS <<EOF || true
dapo-math-17k          | /data/dapo-math-17k/dapo-math-17k.jsonl                                  | rm:math                      |
skywork-or1-math       | /data/skywork-or1-rl/skywork-or1-math-miles-20k.jsonl                    | rm:math                      |
nemotron-math-v2       | /data/nemotron-rl-math-v2/nemotron-rl-math-v2-miles.jsonl                | rm:math                      |
knowledge-mcqa         | /data/nemotron-rl-mcqa/knowledge-mcqa-miles-20k.jsonl                    | rm:gpqa                      |
mmlu-pro               | /data/mmlu-pro/mmlu-pro-miles-2k.jsonl                                   | rm:gpqa                      |
reasoning-gym          | /data/nemotron-rl-reasoninggym/reasoning-gym-miles.jsonl                 | ${NB}.rewards.reasoning_gym_reward      |
structured-outputs     | /data/nemotron-rl-ifollow-struct/structured-outputs-miles.jsonl          | ${NB}.rewards.structured_output_reward  |
instruction-following  | /data/nemotron-rl-ifollow/instruction-following-miles-20k.jsonl          | ${NB}.ifeval_g.ifeval_reward | EXTRA_PIP=nltk+langdetect+immutabledict+absl-py
fncall-pivot           | /data/nemotron-rl-fncall-pivot/fncall-pivot-miles.jsonl                  | ${NB}.rewards.tool_call_match_reward    |
conv-tooluse           | /data/nemotron-rl-conv-tooluse/conv-tooluse-miles-20k.jsonl              | ${NB}.rewards.tool_call_match_reward    |
swe-pivot              | /data/nemotron-rl-swe-pivot/swe-pivot-miles-20k.jsonl                    | ${NB}.rewards.tool_call_match_reward    |
bfcl                   | /data/bfcl/bfcl-ast-miles.jsonl                                          | ${NB}.grid_and_ast.bfcl_ast_reward      |
arc-agi                | /data/nemotron-rl-arc-agi/arc-agi-train-miles.jsonl                      | ${NB}.grid_and_ast.arc_agi_reward       |
competitive-coding     | /data/nemotron-rl-comp-coding/competitive-coding-miles.jsonl             | ${NB}.code_exec.code_exec_reward        |
EOF

# Datasets whose answer is a short string, not a chain of thought. 24k tokens of
# budget is right for math; a tool call or a letter needs a fraction of it, and
# the cap is what bounds wall-clock on a 20k-row sweep.
declare -A MAX_TOKENS=(
    [knowledge-mcqa]=4096  [mmlu-pro]=4096
    [fncall-pivot]=4096    [conv-tooluse]=4096  [swe-pivot]=4096  [bfcl]=4096
    [structured-outputs]=8192  [instruction-following]=8192
    [arc-agi]=16384        [competitive-coding]=16384
)

mode="${1:-list}"
only="${2:-}"

while IFS='|' read -r name data verifier extra; do
    name="$(echo "${name}" | xargs)"; [[ -z "${name}" ]] && continue
    data="$(echo "${data}" | xargs)"
    verifier="$(echo "${verifier}" | xargs)"
    extra="$(echo "${extra}" | xargs)"
    [[ -n "${only}" && "${name}" != "${only}" ]] && continue

    if [[ "${verifier}" == rm:* ]]; then
        rm_export="RM_TYPE=${verifier#rm:}"
    else
        rm_export="RM_TYPE=math,CUSTOM_RM_PATH=${verifier}"
    fi
    tokens="${MAX_TOKENS[${name}]:-24576}"

    case "${mode}" in
    list)
        printf "%-22s %-6s %s\n" "${name}" "${tokens}" "${verifier}"
        continue
        ;;
    smoke)
        job="smoke-${name}"
        out="/data/difficulty/smoke/${name}.${MODEL}.passrate.jsonl"
        # Responses, not just scores: a 0.000 mean has to be readable as
        # "answered wrong" or "verifier never saw a tool call".
        limits="LIMIT=64,DUMP_RESPONSES=/data/difficulty/smoke/${name}.${MODEL}.audit.jsonl,DUMP_LIMIT=8"
        ;;
    full)
        job="pr-${name}"
        out="/data/difficulty/${name}.${MODEL}.passrate.jsonl"
        limits=""
        ;;
    *)
        echo "usage: $0 {list|smoke|full} [dataset]" >&2; exit 2 ;;
    esac

    # --export is comma-separated, so no value here may contain a comma. That is
    # also why EXTRA_PIP uses '+' between package names.
    exports="ALL,MODEL_NAME=${MODEL},PROMPT_DATA=${data},OUTPUT=${out},MAX_NEW_TOKENS=${tokens},PREFLIGHT=strict,${rm_export}"
    [[ -n "${extra}" ]] && exports="${exports},${extra}"
    [[ -n "${limits}" ]] && exports="${exports},${limits}"

    id=$(sbatch --parsable -A "${ACCOUNT}" --job-name="${job}" \
        --export="${exports}" \
        experiments/src/difficulty_filter/run_measure.sbatch)
    printf "%-22s %s  job=%s\n" "${name}" "${mode}" "${id}"
done <<< "${SWEEPS}"
