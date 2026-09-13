import json
import struct

import pytest

from experiments.src.checkpoints.lightning_sft import native_policy_config, stage_policy


def write_tensor(path, name, data):
    header = json.dumps({name: {"dtype": "BF16", "shape": [len(data) // 2], "data_offsets": [0, len(data)]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_stage_preserves_sft_bytes_and_source_and_removes_mtp(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 3,
                "hybrid_override_pattern": "M*E",
                "auto_map": {"AutoConfig": "legacy.Config"},
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*E",
            }
        )
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"backbone.weight": "policy.safetensors", "mtp.weight": "mtp.safetensors"}})
    )
    write_tensor(source / "policy.safetensors", "backbone.weight", b"\x01\x02\x03\x04")
    write_tensor(source / "mtp.safetensors", "mtp.weight", b"\x05\x06")
    (source / "tokenizer.json").write_text("{}")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    report = stage_policy(source, target, source_id="original-sft")
    assert (target / "policy.safetensors").read_bytes() == original["policy.safetensors"]
    assert not (target / "mtp.safetensors").exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    config = json.loads((target / "config.json").read_text())
    assert config["layers_block_type"] == ["mamba", "attention", "moe"]
    assert config["num_nextn_predict_layers"] == 0 and "auto_map" not in config
    assert report["policy_bytes"] == 4 and report["excluded_mtp_tensors"] == 1
    assert stage_policy(source, target, source_id="original-sft") == report
    with pytest.raises(ValueError, match="different source"):
        stage_policy(source, target, source_id="another-sft")
    (target / "policy.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        stage_policy(source, target, source_id="original-sft")


def test_conflicting_hybrid_descriptions_are_rejected():
    with pytest.raises(ValueError, match="disagree"):
        native_policy_config({"num_hidden_layers": 1, "hybrid_override_pattern": "M", "layers_block_type": ["moe"]})
