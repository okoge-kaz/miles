#!/bin/bash
# Delete exactly one derived training-checkpoint directory when explicitly
# requested. This file is sourced after common/run_identity.sh has set
# CKPT_PATH and experiments/env.sh has set TRAIN_CKPT_DIR.

: "${CLEAN_CHECKPOINT:=0}"
[[ "${CLEAN_CHECKPOINT}" =~ ^[01]$ ]] || {
    echo "CLEAN_CHECKPOINT must be 0 or 1" >&2
    return 1
}

if [[ "${CLEAN_CHECKPOINT}" == 1 ]]; then
    [[ "${CKPT_PATH}" == /ckpt/training/* ]] || {
        echo "refusing to clean checkpoint outside /ckpt/training: ${CKPT_PATH}" >&2
        return 1
    }

    checkpoint_relative="${CKPT_PATH#/ckpt/training/}"
    [[ -n "${checkpoint_relative}" ]] || {
        echo "refusing to clean the training-checkpoint root" >&2
        return 1
    }
    case "/${checkpoint_relative}/" in
        *"/../"*|*"/./"*|*"//"*)
            echo "refusing unsafe checkpoint path: ${CKPT_PATH}" >&2
            return 1
            ;;
    esac

    checkpoint_root="${TRAIN_CKPT_DIR%/}"
    host_checkpoint_path="${checkpoint_root}/${checkpoint_relative}"
    [[ "${host_checkpoint_path}" == "${checkpoint_root}/"* ]] || {
        echo "refusing checkpoint path outside ${checkpoint_root}: ${host_checkpoint_path}" >&2
        return 1
    }

    if [[ -e "${host_checkpoint_path}" || -L "${host_checkpoint_path}" ]]; then
        echo "cleaning checkpoint ${host_checkpoint_path}"
        rm -rf -- "${host_checkpoint_path}"
    else
        echo "clean checkpoint requested; path is already absent: ${host_checkpoint_path}"
    fi
fi

unset checkpoint_relative checkpoint_root host_checkpoint_path
