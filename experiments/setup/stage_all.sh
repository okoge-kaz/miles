#!/bin/bash
# Stage every model and dataset listed in models.txt / datasets.txt.
#
#   experiments/setup/stage_all.sh              # models + datasets
#   experiments/setup/stage_all.sh models       # models only
#   experiments/setup/stage_all.sh datasets     # datasets only
#
# Run from the repo root. Submits, then returns; watch with experiments/status.sh.
#
# Scheduling policy:
#   * downloads go to cpu_interactive (highest tier, no GPU held while pulling)
#   * each model's conversion depends on its own download AND on the previous
#     conversion, so only one 8-GPU job of this batch runs at a time
#   * dataset downloads run concurrently, capped by MAX_PARALLEL_DL chains

set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
WHAT="${1:-all}"
MAX_PARALLEL_DL="${MAX_PARALLEL_DL:-4}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

strip() { sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | grep -v '^$'; }

stage_models() {
    local prev_convert=""
    while IFS='|' read -r name repo type extra nodes; do
        name=$(echo "$name" | xargs); repo=$(echo "$repo" | xargs); type=$(echo "$type" | xargs)
        extra=$(echo "${extra:-}" | xargs); nodes=$(echo "${nodes:-1}" | xargs); nodes=${nodes:-1}
        [[ -z "$name" ]] && continue

        local dl_dep="" cv_dep=""
        if [[ -f "${HF_CKPT_DIR:?source experiments/env.sh first}/${name}/.download_complete" ]]; then
            echo "skip download ${name} (complete)"
        else
            dl=$(sbatch --parsable -A "$ACCOUNT" \
                 --job-name="dl-${name}" \
                 --export=ALL,HF_REPO="${repo}",MODEL_NAME="${name}" \
                 experiments/setup/download_model.sbatch)
            dl_dep="afterok:${dl}"
        fi

        local tracker="${MEGATRON_CKPT_DIR:?source experiments/env.sh first}/${name}_torch_dist/latest_checkpointed_iteration.txt"
        if [[ -z "$dl_dep" && -f "$tracker" && "$(cat "$tracker" 2>/dev/null)" == "release" ]]; then
            echo "skip ${name} (already downloaded and converted)"
            continue
        fi

        cv_dep="${dl_dep}"
        [[ -n "$prev_convert" ]] && cv_dep="${cv_dep:+${cv_dep},}afterany:${prev_convert}"

        cv=$(sbatch --parsable -A "$ACCOUNT" \
             --job-name="cv-${name}" \
             ${cv_dep:+--dependency=${cv_dep}} \
             -p interactive --time=04:00:00 --nodes="${nodes}" \
             --export=ALL,MODEL_NAME="${name}",MEGATRON_MODEL_TYPE="${type}",CONVERT_EXTRA_ARGS="${extra}" \
             experiments/setup/convert_checkpoint.sbatch)
        prev_convert="$cv"
        printf "%-40s download=%-10s convert=%s\n" "$name" "${dl:-skipped}" "$cv"
        unset dl
    done < <(strip < experiments/setup/models.txt)
}

stage_datasets() {
    local -a chain_tail=()
    local i=0
    while IFS='|' read -r name repo; do
        name=$(echo "$name" | xargs); repo=$(echo "$repo" | xargs)
        [[ -z "$name" ]] && continue

        local slot=$(( i % MAX_PARALLEL_DL ))
        local dep=""
        [[ -n "${chain_tail[$slot]:-}" ]] && dep="--dependency=afterany:${chain_tail[$slot]}"

        jid=$(sbatch --parsable -A "$ACCOUNT" \
              --job-name="ds-${name}" \
              ${dep} \
              --export=ALL,HF_REPO="${repo}",LOCAL_NAME="${name}" \
              experiments/setup/download_dataset.sbatch)
        chain_tail[$slot]="$jid"
        printf "%-24s %-44s job=%s\n" "$name" "$repo" "$jid"
        i=$(( i + 1 ))
    done < <(strip < experiments/setup/datasets.txt)
}

case "$WHAT" in
    models)   echo "=== models ===";   stage_models ;;
    datasets) echo "=== datasets ==="; stage_datasets ;;
    all)      echo "=== models ===";   stage_models
              echo; echo "=== datasets ==="; stage_datasets ;;
    *) echo "usage: $0 [all|models|datasets]"; exit 1 ;;
esac
