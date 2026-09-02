#!/bin/bash
# Stage ONE model: download the HF weights, then convert them to torch_dist.
#
#   experiments/setup/download/stage_model.sh <MODEL_NAME>
#
# MODEL_NAME is looked up in models.txt, which supplies the HF repo, the
# MODEL_ARGS file and any per-model convert overrides. Use this instead of
# stage_all.sh when only one model is needed: stage_all.sh walks the whole list
# and would submit a conversion for every entry that is not already converted.
#
# Both steps are idempotent and skipped when already done:
#   * download  — skipped if $HF_CKPT_DIR/<name>/.download_complete is nonempty
#   * convert   — skipped if the torch_dist tracker file reads "release"
# The convert job additionally re-checks the tracker itself
# (convert_checkpoint.sbatch), so a race with a concurrent stage is harmless.
#
# Submits and returns; watch with experiments/status.sh.

set -euo pipefail

MODEL_NAME="${1:?usage: stage_model.sh <MODEL_NAME>   (a name from experiments/setup/manifests/models.txt)}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh
source experiments/common/pbs.sh

SETUP_CONVERT_WALLTIME="${SETUP_CONVERT_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"
SETUP_DOWNLOAD_WALLTIME="${SETUP_DOWNLOAD_WALLTIME:-${PBS_DOWNLOAD_WALLTIME:-24:00:00}}"
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"
HF_DOWNLOAD_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-5}"
HF_DOWNLOAD_RETRY_DELAY_SECONDS="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-60}"
[[ "${HF_DOWNLOAD_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_MAX_WORKERS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_ATTEMPTS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || {
    echo "HF_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
}
SETUP_PATH_EXPORTS="MILES_WORKSPACE_ROOT,MILES_REPO,CHECKPOINT_ROOT,HF_CKPT_DIR,MEGATRON_CKPT_DIR,TRAIN_CKPT_DIR,DATASET_ROOT,PRETRAIN_DATASET_DIR,RL_DATASET_DIR,SFT_DATASET_DIR,DATASET_DIR,CONTAINER_DIR,CACHE_DIR,CONTAINER_IMAGE"
MODELS_TXT="experiments/setup/manifests/models.txt"
HF_TOKEN_EXPORT=""
[[ -z "${HF_TOKEN:-}" ]] || HF_TOKEN_EXPORT=",HF_TOKEN"
export HF_DOWNLOAD_MAX_WORKERS HF_DOWNLOAD_ATTEMPTS HF_DOWNLOAD_RETRY_DELAY_SECONDS

# Pull the one row out of models.txt. Comments and blank lines are dropped
# first so a commented-out (disabled) model cannot be staged by accident.
row=$(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "${MODELS_TXT}" \
      | grep -v '^$' \
      | awk -F'|' -v want="${MODEL_NAME}" \
            '{ k = $1; gsub(/^[ \t]+|[ \t]+$/, "", k); if (k == want) print }')

if [[ -z "${row}" ]]; then
    echo "no such model in ${MODELS_TXT}: ${MODEL_NAME}"
    echo "available:"
    sed -e 's/#.*//' "${MODELS_TXT}" | awk -F'|' 'NF>1 { gsub(/^[ \t]+|[ \t]+$/, "", $1); print "  " $1 }'
    exit 1
fi

IFS='|' read -r name repo type extra nodes <<< "${row}"
name=$(echo "${name}" | xargs)
repo=$(echo "${repo}" | xargs)
type=$(echo "${type}" | xargs)
extra=$(echo "${extra:-}" | xargs)
nodes=$(echo "${nodes:-1}" | xargs); nodes=${nodes:-1}

[[ -f "scripts/models/${type}.sh" ]] || { echo "no MODEL_ARGS file: scripts/models/${type}.sh"; exit 1; }

echo "model   ${name}"
echo "repo    ${repo}"
echo "args    scripts/models/${type}.sh"
echo "extra   ${extra:-none}"
echo "nodes   ${nodes}"
echo

dl_dep=""
if [[ -s "${HF_CKPT_DIR}/${name}/.download_complete" ]]; then
    echo "skip download: ${HF_CKPT_DIR}/${name} already complete"
else
    dl=$(pbs_submit --parsable --profile=cpu \
         --job-name="dl-${name}" \
         --time="${SETUP_DOWNLOAD_WALLTIME}" \
         --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,HF_DOWNLOAD_MAX_WORKERS,HF_DOWNLOAD_ATTEMPTS,HF_DOWNLOAD_RETRY_DELAY_SECONDS,HF_REPO=${repo},MODEL_NAME=${name}${HF_TOKEN_EXPORT}" \
         experiments/setup/download/download_model.sbatch)
    dl_dep="--dependency=afterok:${dl}"
    echo "download  ${dl}"
fi

tracker="${MEGATRON_CKPT_DIR}/${name}_torch_dist/latest_checkpointed_iteration.txt"
if [[ -z "${dl_dep}" && -f "${tracker}" && "$(cat "${tracker}" 2>/dev/null)" == "release" ]]; then
    echo "skip convert: ${name}_torch_dist already at release"
    exit 0
fi

cv=$(pbs_submit --parsable --profile=gpu \
     --job-name="cv-${name}" \
     ${dl_dep} \
     --time="${SETUP_CONVERT_WALLTIME}" --nodes="${nodes}" \
     --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,MODEL_NAME=${name},MEGATRON_MODEL_TYPE=${type},CONVERT_EXTRA_ARGS=${extra}" \
     experiments/setup/models/convert_checkpoint.sbatch)
echo "convert   ${cv}${dl_dep:+  (waits on ${dl})}"
echo
echo "  ${HF_CKPT_DIR}/${name}"
echo "  ${MEGATRON_CKPT_DIR}/${name}_torch_dist"
