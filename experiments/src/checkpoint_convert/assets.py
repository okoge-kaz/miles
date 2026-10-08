import shutil
from pathlib import Path


def copy_hf_assets(source: Path, destination: Path) -> None:
    """Preserve HF metadata without replacing the converted weights or index."""
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    weight_indexes = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    for asset in source.iterdir():
        if not asset.is_file() or asset.name in weight_indexes or asset.name.endswith(weight_suffixes):
            continue
        shutil.copy2(asset, destination / asset.name)
