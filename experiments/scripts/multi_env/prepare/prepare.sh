#!/bin/bash

set -euo pipefail

# IF uses the pinned download; Tau2 uses native data already baked into the SIF.
: "${DATA_ROOT:=/data}"
: "${OUTPUT_ROOT:=${DATA_ROOT}/multi_env}"
# Keep the existing download location; the source is IF training, not IFBench test.
: "${IF_TRAIN_SOURCE:=${DATA_ROOT}/ifbench-train}"
: "${DATASET_NUM_PROC:=1}"
: "${DATASET_CHUNK_SIZE:=128}"

python3 -m experiments.src.datasets.instruction_following.download --output "${IF_TRAIN_SOURCE}"
python3 -m experiments.src.datasets.instruction_following.prepare \
    --input "${IF_TRAIN_SOURCE}/data/train-00000-of-00001.parquet" --output "${OUTPUT_ROOT}/if-training.jsonl" \
    --num-proc "${DATASET_NUM_PROC}" --chunk-size "${DATASET_CHUNK_SIZE}"

: "${NEMO_GYM_ROOT:=/opt/nemo-gym}"
NEMO_GYM_OFFLINE=1 "${NEMO_GYM_ROOT}/.venv/bin/python" -m experiments.src.datasets.tau2.prepare \
    --input "${NEMO_GYM_ROOT}/benchmarks/tau2/data/tau2_benchmark.jsonl" \
    --train-output "${OUTPUT_ROOT}/tau2-train.jsonl" \
    --validation-output "${OUTPUT_ROOT}/tau2-test.jsonl" \
    --num-proc "${DATASET_NUM_PROC}" --chunk-size "${DATASET_CHUNK_SIZE}"
