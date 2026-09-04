#!/bin/bash
# Remove an explicitly selected DAPO math async checkpoint cohort that used
# TBQ=1000 at staleness >= 8.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

DELETE=0
COHORT=""
usage() {
    cat >&2 <<'EOF'
usage: experiments/cleanup_tbq1000_staleness_checkpoints.sh \
           --cohort truncation-ablations|baseline-s8|all [--delete]

The default is a dry run. truncation-ablations selects the six zero-loss arms
and four async no-truncation-treatment arms. baseline-s8 selects the completed
zero-reward s8 arms at ratios 1:7, 2:6, 3:5, and 4:4. all selects both cohorts.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --cohort)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            COHORT="$2"
            shift 2
            ;;
        --delete)
            DELETE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ "${COHORT}" =~ ^(truncation-ablations|baseline-s8|all)$ ]] || {
    echo "--cohort must be truncation-ablations, baseline-s8, or all" >&2
    usage
    exit 2
}

declare -a checkpoint_names=()
if [[ "${COHORT}" =~ ^(baseline-s8|all)$ ]]; then
    for ratio in t1r7 t2r6 t3r5 t4r4; do
        checkpoint_names+=("s8-${ratio}-sr-20260819-212906-zero-trunc-rb-inflight")
    done
fi
if [[ "${COHORT}" =~ ^(truncation-ablations|all)$ ]]; then
    for staleness in 8 16 20; do
        for ratio in t1r7 t2r6; do
            checkpoint_names+=(
                "s${staleness}-${ratio}-hiso-zero-loss-trunc-s8-16-20-r12-20260827-v1-zero-loss-trunc-rb-inflight"
            )
        done
    done
    for staleness in 8 16; do
        for ratio in t1r7 t2r6; do
            checkpoint_names+=(
                "s${staleness}-${ratio}-hiso-reward-off-trunc-coloc-s8-16-r12-20260827-v1-zero-reward-trunc-off-rb-inflight"
            )
        done
    done
fi

declare -a checkpoint_paths=()
for checkpoint_name in "${checkpoint_names[@]}"; do
    mapfile -t matches < <(
        find "${TRAIN_CKPT_DIR}" \
            -type d \
            -name "${checkpoint_name}" \
            -print
    )
    (( ${#matches[@]} == 1 )) || {
        echo "expected exactly one checkpoint named ${checkpoint_name}, found ${#matches[@]}" >&2
        exit 1
    }
    checkpoint_path="${matches[0]}"
    [[ "${checkpoint_path}" == "${TRAIN_CKPT_DIR}"/math/*/async/off-policy/max-weight-staleness-*/* ]] || {
        echo "refusing checkpoint outside the expected async math tree: ${checkpoint_path}" >&2
        exit 1
    }
    [[ "${checkpoint_path}" != *-tbq6000 ]] || {
        echo "refusing to remove a corrected TBQ=6000 checkpoint: ${checkpoint_path}" >&2
        exit 1
    }
    checkpoint_paths+=("${checkpoint_path}")
done

printf 'identified %s TBQ=1000 checkpoints in cohort %s:\n' \
    "${#checkpoint_paths[@]}" "${COHORT}"
printf '  %s\n' "${checkpoint_paths[@]}"

if (( DELETE == 0 )); then
    printf '\ndry run; add --delete to remove exactly these checkpoints\n'
    exit 0
fi

for checkpoint_path in "${checkpoint_paths[@]}"; do
    rm -rf --one-file-system -- "${checkpoint_path}"
    printf 'deleted %s\n' "${checkpoint_path}"
done
