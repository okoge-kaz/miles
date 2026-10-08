import json

import torch
from safetensors.torch import load_file, save_file

from experiments.src.checkpoint_convert.assets import copy_hf_assets


def test_bridge_export_preserves_tokenizer_without_overwriting_weights(tmp_path):
    source, converted = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    converted.mkdir()
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "add_bos_token": False}
    special_tokens = {"eos_token": "</s>"}
    (source / "tokenizer_config.json").write_text(json.dumps(config))
    (source / "special_tokens_map.json").write_text(json.dumps(special_tokens))
    (converted / "tokenizer_config.json").write_text('{"tokenizer_class": "TokenizersBackend"}')
    save_file({"weight": torch.zeros(2)}, source / "model.safetensors")
    save_file({"weight": torch.ones(2)}, converted / "model.safetensors")
    index = '{"weight_map": {"weight": "converted.safetensors"}}'
    (converted / "model.safetensors.index.json").write_text(index)
    (source / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (source / "pytorch_model.bin").write_bytes(b"unconverted weights")

    copy_hf_assets(source, converted)

    assert json.loads((converted / "tokenizer_config.json").read_text()) == config
    assert json.loads((converted / "special_tokens_map.json").read_text()) == special_tokens
    torch.testing.assert_close(load_file(converted / "model.safetensors")["weight"], torch.ones(2))
    assert (converted / "model.safetensors.index.json").read_text() == index
    assert not (converted / "pytorch_model.bin").exists()
