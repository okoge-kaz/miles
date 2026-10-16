import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class _TensorRef:
    path: Path
    name: str
    expert: int | None = None
    half: int | None = None

    def read(self, handle) -> torch.Tensor:
        tensor = handle.get_tensor(self.name)
        if self.expert is not None:
            tensor = tensor[self.expert]
        if self.half is not None:
            tensor = tensor.chunk(2, dim=0)[self.half]
        return tensor


def _snapshot(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        item = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        if path.suffix != ".safetensors":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[path.name] = item
    return result


def _weight_map(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate tensor: {name}")
                result[name] = path
    if not result:
        raise ValueError(f"no safetensors weights: {directory}")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        declared = json.loads(index.read_text())["weight_map"]
        if declared != {name: path.name for name, path in result.items()}:
            raise ValueError(f"weight index does not match actual tensors: {directory}")
    return result


def _logical_weight_map(encoded: dict[str, Path]) -> dict[str, _TensorRef]:
    result = {}
    for name, path in encoded.items():
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            prefix, projection = name.rsplit(".", 1)
            with safe_open(path, framework="pt", device="cpu") as handle:
                shape = handle.get_slice(name).get_shape()
            if len(shape) != 3 or (projection == "gate_up_proj" and shape[1] % 2):
                raise ValueError(f"invalid fused expert tensor: {name}, shape={shape}")
            projections = ("gate", "up") if projection == "gate_up_proj" else ("down",)
            for expert in range(shape[0]):
                for half, output_projection in enumerate(projections):
                    key = f"{prefix}.{expert}.{output_projection}_proj.weight"
                    ref = _TensorRef(path, name, expert=expert, half=half if len(projections) == 2 else None)
                    if key in result:
                        raise ValueError(f"duplicate logical tensor: {key}")
                    result[key] = ref
        else:
            if name in result:
                raise ValueError(f"duplicate logical tensor: {name}")
            result[name] = _TensorRef(path, name)
    return result


def _resolve_tied_head(directory: Path, weights: dict[str, _TensorRef]) -> dict[str, str]:
    config_file = directory / "config.json"
    if not config_file.is_file():
        return {}
    config = json.loads(config_file.read_text())
    tied = config.get("tie_word_embeddings")
    if tied is None:
        tied = config.get("text_config", {}).get("tie_word_embeddings", False)
    if not tied or "lm_head.weight" in weights:
        return {}
    for name in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"):
        if name in weights:
            weights["lm_head.weight"] = weights[name]
            return {"lm_head.weight": name}
    return {}


def _tensor_difference(source: torch.Tensor, converted: torch.Tensor, *, atol: float, rtol: float) -> dict:
    if source.shape != converted.shape:
        return {"passed": False, "source_shape": list(source.shape), "converted_shape": list(converted.shape)}
    # Bound temporary FP32 comparisons even for the embedding/vocabulary tensor.
    source_flat, converted_flat = source.reshape(-1), converted.reshape(-1)
    mismatch_count, max_abs_error, all_finite = 0, 0.0, True
    for start in range(0, source.numel(), 4 * 1024**2):
        left = source_flat[start : start + 4 * 1024**2].float()
        right = converted_flat[start : start + 4 * 1024**2].float()
        finite = torch.isfinite(left).all() and torch.isfinite(right).all()
        all_finite = all_finite and bool(finite)
        error = (left - right).abs()
        max_abs_error = max(max_abs_error, float(error.max().item()))
        mismatch_count += int((error > atol + rtol * left.abs()).sum().item())
    return {
        "passed": all_finite and mismatch_count == 0,
        "elements": source.numel(),
        "source_dtype": str(source.dtype),
        "converted_dtype": str(converted.dtype),
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
        "all_finite": all_finite,
    }


def _compare_weights(source: Path, converted: Path, *, atol: float, rtol: float) -> dict:
    source_encoded, converted_encoded = _weight_map(source), _weight_map(converted)
    source_map, converted_map = _logical_weight_map(source_encoded), _logical_weight_map(converted_encoded)
    tied_aliases = {
        "source": _resolve_tied_head(source, source_map),
        "converted": _resolve_tied_head(converted, converted_map),
    }
    missing = sorted(source_map.keys() - converted_map.keys())
    extra = sorted(converted_map.keys() - source_map.keys())
    failures, dtype_changes = {}, {}
    tensors, elements, max_abs_error = 0, 0, 0.0
    for name in sorted(source_map.keys() & converted_map.keys()):
        source_ref, converted_ref = source_map[name], converted_map[name]
        with safe_open(source_ref.path, framework="pt", device="cpu") as left:
            with safe_open(converted_ref.path, framework="pt", device="cpu") as right:
                result = _tensor_difference(source_ref.read(left), converted_ref.read(right), atol=atol, rtol=rtol)
        if not result["passed"]:
            failures[name] = result
        if result.get("source_dtype") != result.get("converted_dtype"):
            dtype_changes[name] = [result.get("source_dtype"), result.get("converted_dtype")]
        tensors += 1
        elements += result.get("elements", 0)
        max_abs_error = max(max_abs_error, result.get("max_abs_error", 0.0))
        if tensors % 250 == 0:
            print(f"Compared {tensors} tensors", flush=True)
    return {
        "passed": not missing and not extra and not failures,
        "tensors": tensors,
        "source_encoded_tensors": len(source_encoded),
        "converted_encoded_tensors": len(converted_encoded),
        "tied_weight_aliases": tied_aliases,
        "elements": elements,
        "missing_tensors": missing,
        "extra_tensors": extra,
        "failures": failures,
        "dtype_changes": dtype_changes,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
    }


def _compare_assets(source: Path, converted: Path) -> dict:
    configs = []
    for path in (source, converted):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True).to_dict()
        for key in ("_name_or_path", "transformers_version"):
            config.pop(key, None)
        configs.append(config)
    mismatches = {}
    for name in TOKENIZER_ASSETS:
        original = source / name
        if not original.is_file():
            continue
        output = converted / name
        if not output.is_file():
            mismatches[name] = "missing"
        elif name.endswith(".json"):
            if json.loads(original.read_text()) != json.loads(output.read_text()):
                mismatches[name] = "JSON values differ"
        elif original.read_bytes() != output.read_bytes():
            mismatches[name] = "bytes differ"
    config_changes = {
        key: [configs[0].get(key), configs[1].get(key)]
        for key in configs[0].keys() | configs[1].keys()
        if configs[0].get(key) != configs[1].get(key)
    }
    return {
        "passed": configs[0] == configs[1] and not mismatches,
        "config_equal": configs[0] == configs[1],
        "config_changes": config_changes,
        "tokenizer_mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.snapshot:
        args.snapshot.write_text(json.dumps(_snapshot(args.source), indent=2) + "\n")
        return
    if not args.converted or not args.before or not args.report:
        parser.error("comparison requires --converted, --before and --report")
    weights = _compare_weights(args.source, args.converted, atol=args.atol, rtol=args.rtol)
    assets = _compare_assets(args.source, args.converted)
    unchanged = json.loads(args.before.read_text()) == _snapshot(args.source)
    report = {
        "source": str(args.source),
        "converted": str(args.converted),
        "passed": weights["passed"] and assets["passed"] and unchanged,
        "weights": weights,
        "assets": assets,
        "source_unchanged": unchanged,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "tensors": weights["tensors"],
                "max_abs_error": weights["max_abs_error"],
                "source_unchanged": unchanged,
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit("HF roundtrip comparison failed; inspect the full report")


if __name__ == "__main__":
    main()
