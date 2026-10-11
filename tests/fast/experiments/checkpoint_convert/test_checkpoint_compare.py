from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
pytest.importorskip("transformers")

from experiments.src.checkpoint_convert.compare import _compare_weights, _tensor_difference


def _write_checkpoint(directory: Path, weights: dict) -> None:
    directory.mkdir()
    safetensors.save_file(weights, str(directory / "model.safetensors"))


def test_full_comparison_accepts_equivalent_fused_experts_and_checks_vision_and_mtp(tmp_path):
    source, exported = tmp_path / "source", tmp_path / "exported"
    fused = torch.arange(48, dtype=torch.bfloat16).reshape(2, 6, 4)
    down = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3)
    prefix = "model.language_model.layers.0.mlp.experts"
    mtp = torch.arange(4, dtype=torch.float32)
    vision = torch.arange(8, dtype=torch.bfloat16)
    _write_checkpoint(
        source,
        {
            f"{prefix}.gate_up_proj": fused,
            f"{prefix}.down_proj": down,
            "mtp.norm.weight": mtp,
            "model.visual.weight": vision,
        },
    )
    weights = {"mtp.norm.weight": mtp, "model.visual.weight": vision}
    for expert in range(2):
        gate, up = fused[expert].chunk(2, dim=0)
        weights.update(
            {
                f"{prefix}.{expert}.gate_proj.weight": gate.contiguous(),
                f"{prefix}.{expert}.up_proj.weight": up.contiguous(),
                f"{prefix}.{expert}.down_proj.weight": down[expert].contiguous(),
            }
        )
    _write_checkpoint(exported, weights)
    result = _compare_weights(source, exported, atol=1e-6, rtol=1e-5)
    assert result["passed"] and result["max_abs_error"] == 0
    assert result["elements"] == 84 and result["tensors"] == 8

    # A successful text-weight conversion must not hide omitted auxiliary weights.
    del weights["model.visual.weight"]
    weights["mtp.norm.weight"] = mtp + 1
    safetensors.save_file(weights, str(exported / "model.safetensors"))
    result = _compare_weights(source, exported, atol=1e-6, rtol=1e-5)
    assert not result["passed"]
    assert result["missing_tensors"] == ["model.visual.weight"]
    assert result["failures"]["mtp.norm.weight"]["mismatch_count"] == 4


def test_comparison_detects_corruption_after_first_chunk_and_nonfinite_values():
    original = torch.zeros(4 * 1024**2 + 2)
    converted = original.clone()
    converted[-1] = 1
    result = _tensor_difference(original, converted, atol=1e-6, rtol=1e-5)
    assert not result["passed"] and result["mismatch_count"] == 1
    converted[-1] = float("nan")
    result = _tensor_difference(original, converted, atol=1e-6, rtol=1e-5)
    assert not result["passed"] and not result["all_finite"]


@pytest.mark.parametrize("tie", [True, False])
def test_redundant_output_head_is_allowed_only_for_tied_model_and_equal_weights(tmp_path, tie):
    source, exported = tmp_path / "source", tmp_path / "exported"
    embedding = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    _write_checkpoint(source, {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding.clone()})
    _write_checkpoint(exported, {"model.embed_tokens.weight": embedding})
    for directory in (source, exported):
        (directory / "config.json").write_text(json.dumps({"tie_word_embeddings": tie}))
    result = _compare_weights(source, exported, atol=1e-6, rtol=1e-5)
    assert result["passed"] == tie
    if tie:
        safetensors.save_file(
            {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding + 1},
            str(source / "model.safetensors"),
        )
        result = _compare_weights(source, exported, atol=1e-6, rtol=1e-5)
        assert not result["passed"] and "lm_head.weight" in result["failures"]
