from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.search_r1.evaluation.summarize import build_summary, summarize_records
from experiments.src.datasets.search_r1.build_eval import PROMPT_TEMPLATE, convert_row
from experiments.src.environments.search_r1.retrieval_server import SearchRequest, search_batch


REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeEncoder:
    def __call__(self, queries: list[str]) -> np.ndarray:
        return np.ones((len(queries), 2), dtype=np.float32)


class _FakeIndex:
    def search(self, embeddings: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        row_count = len(embeddings)
        scores = np.tile(np.arange(topk, 0, -1, dtype=np.float32), (row_count, 1))
        ids = np.tile(np.arange(topk, dtype=np.int64), (row_count, 1))
        return scores, ids


def test_build_eval_matches_training_prompt_and_reward_schema() -> None:
    converted = convert_row(
        {
            "id": "nq-1",
            "question": "Who won?",
            "golden_answers": ["Alice", "A. Example"],
        }
    )

    assert converted == {
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question="Who won?")}],
        "reward_model": {
            "ground_truth": {"target": ["Alice", "A. Example"]},
            "style": "rule",
        },
        "metadata": {"source": "nq-1", "question": "Who won?"},
    }


def test_retrieval_batches_requests_without_mixing_response_boundaries() -> None:
    requests = [
        SearchRequest(queries=("first", "second"), topk=2, return_scores=False),
        SearchRequest(queries=("third",), topk=1, return_scores=True),
    ]
    responses = search_batch(
        _FakeIndex(),
        titles=["zero", "one"],
        texts=["body zero", "body one"],
        encoder=_FakeEncoder(),
        requests=requests,
    )

    assert [len(group) for group in responses[0]["result"]] == [2, 2]
    assert responses[0]["result"][0][0] == {
        "document": {"contents": '"zero"\nbody zero'},
    }
    assert responses[1]["result"] == [
        [{"document": {"contents": '"zero"\nbody zero'}, "score": 2.0}]
    ]


def test_search_r1_evaluation_is_interactive_and_offline() -> None:
    launcher = (REPO_ROOT / "experiments/search_r1/evaluation/run.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --qos=interactive" in launcher
    assert 'WANDB_MODE="${WANDB_MODE:-offline}"' in launcher
    assert '[[ "${WANDB_MODE}" == offline ]]' in launcher
    assert "experiments/search_r1/common/run_measurement.sh" in launcher
    assert "CHECKPOINT_MANIFEST_SHA256" in launcher
    assert "evaluation-contract.env" in launcher
    assert "${MODEL_RUNTIME_DIR}:/search-eval-model" in launcher
    assert "${RESULT_ROOT_HOST}:/results" in launcher
    assert "Qwen3-4B-Instruct-2507" not in launcher
    assert "Qwen3-4B-Base-LR2e-5-Step4000" in launcher
    assert "iter_0004000" in launcher

    runtime = (REPO_ROOT / "experiments/search_r1/common/run_measurement.sh").read_text(
        encoding="utf-8"
    )
    assert "pip install" not in runtime
    assert "artifact-manifest.sha256" in runtime
    assert 'touch "${RESULT_ROOT}/_SUCCESS"' in runtime


def test_search_r1_faiss_is_baked_and_checksum_pinned() -> None:
    derivation = (
        REPO_ROOT / "experiments/container/derive_search_r1_image.sbatch"
    ).read_text(encoding="utf-8")

    assert "faiss-cpu" in derivation
    assert "1.12.0" in derivation
    assert "c2e4963c7188f57cfba248f09ebd8a14c76b5ffb87382603ccd4576f2da39d74" in derivation
    assert "pip install --no-deps" in derivation
    assert "IndexFlatIP" in derivation
    assert '[[ ! -e "${OUTPUT_IMAGE}" ]]' in derivation
    assert '"${OUTPUT_IMAGE}.provenance.env"' in derivation
    assert 'install -m 0444 "${WHEEL_PATH}"' in derivation
    assert "\n    --mount " not in derivation
    assert "grep -E '^(faiss|fastapi|uvicorn)='" in derivation
    assert "/tmp/faiss_cpu.whl" not in derivation


def test_search_r1_filter_is_bound_to_the_sft_policy() -> None:
    launcher = (
        REPO_ROOT / "experiments/tools/difficulty_filter/run_measure_search_r1.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=batch" in launcher
    assert "#SBATCH --qos=interactive" in launcher
    assert "#SBATCH --time=04:00:00" in launcher
    assert "Qwen3-4B-Instruct-2507" not in launcher
    assert "Qwen3-4B-Base-LR2e-5-Step4000" in launcher
    assert "iter_0004000" in launcher
    assert "p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl" in launcher


def test_summarize_records_weights_episode_metrics() -> None:
    records = [
        {
            "n_samples": 1,
            "n_correct": 1,
            "pass_rate": 1.0,
            "response_len_mean": 10.0,
            "observation_len_mean": 20.0,
            "total_response_len_mean": 30.0,
            "truncated_frac": 0.0,
            "search_calls_mean": 1.0,
            "turns_mean": 2.0,
            "searched_frac": 1.0,
            "answered_frac": 1.0,
        },
        {
            "n_samples": 3,
            "n_correct": 0,
            "pass_rate": 0.0,
            "response_len_mean": 20.0,
            "observation_len_mean": 40.0,
            "total_response_len_mean": 60.0,
            "truncated_frac": 1.0,
            "search_calls_mean": 3.0,
            "turns_mean": 3.0,
            "searched_frac": 1.0,
            "answered_frac": 0.0,
        },
    ]

    summary = summarize_records(records)

    assert summary["prompts"] == 2
    assert summary["trajectories"] == 4
    assert summary["exact_match"] == 0.25
    assert summary["prompt_mean_exact_match"] == 0.5
    assert summary["search_calls_mean"] == 2.5
    assert summary["answered_frac"] == 0.25


def test_build_summary_reports_per_benchmark_and_macro(tmp_path: Path) -> None:
    for benchmark, correct in (("nq", 1), ("hotpotqa", 0)):
        benchmark_root = tmp_path / benchmark
        benchmark_root.mkdir()
        record = {
            "n_samples": 1,
            "n_correct": correct,
            "pass_rate": float(correct),
            "response_len_mean": 1.0,
            "observation_len_mean": 2.0,
            "total_response_len_mean": 3.0,
            "truncated_frac": 0.0,
            "search_calls_mean": 1.0,
            "turns_mean": 1.0,
            "searched_frac": 1.0,
            "answered_frac": 1.0,
        }
        (benchmark_root / "records.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    summary = build_summary(tmp_path, ["nq", "hotpotqa"])

    assert summary["benchmark_count"] == 2
    assert summary["macro_exact_match"] == 0.5
    assert summary["benchmarks"]["nq"]["exact_match"] == 1.0
