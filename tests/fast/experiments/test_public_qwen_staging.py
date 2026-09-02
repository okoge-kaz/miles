from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFEST = REPO_ROOT / "experiments/setup/manifests/models.txt"


def _active_manifest_rows() -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for line in MODEL_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        fields = tuple(field.strip() for field in line.split("|"))
        rows[fields[0]] = fields
    return rows


def test_public_qwen3_4b_has_canonical_staging_identity() -> None:
    row = _active_manifest_rows()["Qwen3-4B"]

    assert row == ("Qwen3-4B", "Qwen/Qwen3-4B", "qwen3-4B")
    assert (REPO_ROOT / "scripts/models/qwen3-4B.sh").is_file()


def test_public_qwen3_4b_staging_paths_and_command_are_documented() -> None:
    stage_script = (
        REPO_ROOT / "experiments/setup/download/stage_model.sh"
    ).read_text(encoding="utf-8")
    readme = (REPO_ROOT / "experiments/README.md").read_text(encoding="utf-8")

    assert '${HF_CKPT_DIR}/${name}' in stage_script
    assert '${MEGATRON_CKPT_DIR}/${name}_torch_dist' in stage_script
    assert "stage_model.sh Qwen3-4B" in readme
    assert "checkpoints/hf/Qwen3-4B" in readme
    assert "checkpoints/megatron/Qwen3-4B_torch_dist" in readme


def test_model_download_has_bounded_parallel_retry_controls() -> None:
    stage_script = (
        REPO_ROOT / "experiments/setup/download/stage_model.sh"
    ).read_text(encoding="utf-8")
    worker = (
        REPO_ROOT / "experiments/setup/download/download_model.sbatch"
    ).read_text(encoding="utf-8")

    for name, default in (
        ("HF_DOWNLOAD_MAX_WORKERS", "2"),
        ("HF_DOWNLOAD_ATTEMPTS", "5"),
        ("HF_DOWNLOAD_RETRY_DELAY_SECONDS", "60"),
    ):
        assert f'${{{name}:-{default}}}' in stage_script
        assert name in stage_script.partition("--export=")[2]
        assert f'${{{name}:-{default}}}' in worker

    assert '--max-workers "${HF_DOWNLOAD_MAX_WORKERS}"' in worker
    assert "attempt <= HF_DOWNLOAD_ATTEMPTS" in worker


def test_model_download_completion_marker_is_nonempty_and_atomic() -> None:
    stage_script = (
        REPO_ROOT / "experiments/setup/download/stage_model.sh"
    ).read_text(encoding="utf-8")
    worker = (
        REPO_ROOT / "experiments/setup/download/download_model.sbatch"
    ).read_text(encoding="utf-8")

    assert '[[ -s "${HF_CKPT_DIR}/${name}/.download_complete" ]]' in stage_script
    assert '.download_complete"' in worker
    assert '.partial-${PBS_JOBID:-$$}' in worker
    assert '[[ -s "${_completion_partial}" ]]' in worker
    assert 'mv -f -- "${_completion_partial}" "${_completion_marker}"' in worker
