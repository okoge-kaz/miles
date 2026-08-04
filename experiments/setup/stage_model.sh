#!/bin/bash
# Stage ONE model: download the HF weights, then convert them to torch_dist.
#
#   experiments/setup/stage_model.sh <MODEL_NAME>
#
# MODEL_NAME is looked up in models.txt, which supplies the HF repo, the
# MODEL_ARGS file and any per-model convert overrides. Use this instead of
# stage_all.sh when only one model is needed: stage_all.sh walks the whole list
# and would submit a conversion for every entry that is not already converted.
#
# Both steps are idempotent and skipped when already done:
#   * download  — skipped if $HF_CKPT_DIR/<name>/.download_complete exists
#   * convert   — skipped if the torch_dist tracker file reads "release"
# The convert job additionally re-checks the tracker itself
# (convert_checkpoint.sbatch), so a race with a concurrent stage is harmless.
#
# Submits and returns; watch with experiments/status.sh.

set -euo pipefail

MODEL_NAME="${1:?usage: stage_model.sh <MODEL_NAME>   (a name from experiments/setup/models.txt)}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
MODELS_TXT="experiments/setup/models.txt"

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
if [[ -f "${HF_CKPT_DIR}/${name}/.download_complete" ]]; then
    echo "skip download: ${HF_CKPT_DIR}/${name} already complete"
else
    dl=$(sbatch --parsable -A "${ACCOUNT}" \
         --job-name="dl-${name}" \
         --export=ALL,HF_REPO="${repo}",MODEL_NAME="${name}" \
         experiments/setup/download_model.sbatch)
    dl_dep="--dependency=afterok:${dl}"
    echo "download  ${dl}"
fi

tracker="${MEGATRON_CKPT_DIR}/${name}_torch_dist/latest_checkpointed_iteration.txt"
if [[ -z "${dl_dep}" && -f "${tracker}" && "$(cat "${tracker}" 2>/dev/null)" == "release" ]]; then
    echo "skip convert: ${name}_torch_dist already at release"
    exit 0
fi

cv=$(sbatch --parsable -A "${ACCOUNT}" \
     --job-name="cv-${name}" \
     ${dl_dep} \
     -p interactive --time=04:00:00 --nodes="${nodes}" \
     --export=ALL,MODEL_NAME="${name}",MEGATRON_MODEL_TYPE="${type}",CONVERT_EXTRA_ARGS="${extra}" \
     experiments/setup/convert_checkpoint.sbatch)
echo "convert   ${cv}${dl_dep:+  (waits on ${dl})}"
echo
echo "  ${HF_CKPT_DIR}/${name}"
echo "  ${MEGATRON_CKPT_DIR}/${name}_torch_dist"
