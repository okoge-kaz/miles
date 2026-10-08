#!/bin/bash

# Source after a model preset declares its HF refs and architecture profile.
: "${MODEL_VARIANT:=rl}"
case "${MODEL_VARIANT}" in
    base)
        export MODEL_NAME="${MODEL_NAME:-${HF_BASE_REF##*/}}"
        export HF_MODEL_NAME="${HF_MODEL_NAME:-${HF_BASE_REF##*/}}"
        ;;
    rl)
        export MODEL_NAME="${MODEL_NAME:-${HF_RL_REF##*/}}"
        export HF_MODEL_NAME="${HF_MODEL_NAME:-${HF_RL_REF##*/}}"
        ;;
    sft)
        : "${HF_SFT_REF:?set HF_SFT_REF to an absolute host checkpoint path}"
        [[ "${HF_SFT_REF}" == /* ]] || { echo "HF_SFT_REF must be an absolute host path" >&2; return 2; }
        _hf_root="${HF_CKPT_DIR:-${CHECKPOINT_ROOT:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/checkpoints}/hf}"
        [[ "${HF_SFT_REF}" == "${_hf_root%/}/"* ]] || {
            echo "HF_SFT_REF must be under the HF_CKPT_DIR container mount: ${_hf_root}" >&2; return 2
        }
        export MODEL_NAME="${MODEL_NAME:-${MODEL_SFT_NAME:-${HF_RL_REF##*/}-SFT}}"
        export HF_MODEL_NAME="${HF_MODEL_NAME:-${HF_SFT_REF#"${_hf_root%/}/"}}"
        unset _hf_root
        ;;
    *) echo "MODEL_VARIANT must be base, rl, or sft" >&2; return 2 ;;
esac
export MODEL_VARIANT
