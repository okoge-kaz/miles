from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from experiments.tools.reasoning_eval.validate_checkpoint import (
    checkpoint_manifest_sha256,
    select_latest_checkpoint,
    validate_checkpoint,
)


def _write_safetensors(path: Path, *, include_unindexed_tensor: bool = False) -> None:
    embedding_bytes = 151936 * 4
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [151936, 1],
            "data_offsets": [0, embedding_bytes],
        }
    }
    data = b"\0" * embedding_bytes
    if include_unindexed_tensor:
        header["unexpected.weight"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [embedding_bytes, embedding_bytes + 4],
        }
        data += b"\0" * 4
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + data)


def _write_checkpoint(path: Path, *, include_unindexed_tensor: bool = False) -> Path:
    path.mkdir()
    config = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "generation_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    index = {"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    _write_safetensors(
        path / "model.safetensors",
        include_unindexed_tensor=include_unindexed_tensor,
    )
    return path


def test_checkpoint_manifest_covers_weights_and_tokenizer_files(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")

    info = validate_checkpoint(checkpoint)
    before = checkpoint_manifest_sha256(checkpoint)
    (checkpoint / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
    after = checkpoint_manifest_sha256(checkpoint)

    assert info.shard_count == 1
    assert info.tensor_count == 1
    assert len(before) == 64
    assert after != before


def test_checkpoint_rejects_unindexed_shard_tensors(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint",
        include_unindexed_tensor=True,
    )

    with pytest.raises(ValueError, match="unindexed tensors"):
        validate_checkpoint(checkpoint)


def test_checkpoint_rejects_shard_path_traversal(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    index = {"weight_map": {"model.embed_tokens.weight": "../model.safetensors"}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid shard names"):
        validate_checkpoint(checkpoint)


def test_checkpoint_rejects_truncated_shard(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])

    with pytest.raises(ValueError, match="incomplete or oversized shard"):
        validate_checkpoint(checkpoint)


def test_latest_checkpoint_selection_is_numeric_and_skips_incomplete_exports(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "hf"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root / "2")
    _write_checkpoint(checkpoint_root / "9")
    (checkpoint_root / "10").mkdir()
    (checkpoint_root / "not-a-step").mkdir()

    assert select_latest_checkpoint(checkpoint_root) == (checkpoint_root / "9").resolve()


def test_post_training_wrapper_uses_validated_selection_and_atomic_record() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    wrapper = (
        repo_root
        / "experiments"
        / "scripts"
        / "reasoning_eval"
        / "run-after-training.sbatch"
    ).read_text(encoding="utf-8")

    assert "--latest-under" in wrapper
    assert "selected_checkpoint_partial" in wrapper
    assert 'cmp --silent "${selected_checkpoint_partial}"' in wrapper
