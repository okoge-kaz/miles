import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from miles.utils.dataset_mixture_config import read_prompt_config


@dataclass(frozen=True)
class PromptSource:
    name: str
    path: str
    ratio: float | None = None
    category: str | None = None
    generate_function_path: str | None = None
    rm_type: str | None = None
    custom_rm_path: str | None = None
    input_key: str | None = None
    label_key: str | None = None
    metadata_key: str | None = None
    tool_key: str | None = None


def prompt_sources(args) -> tuple[list[PromptSource], int | None]:
    files = getattr(args, "prompt_data_files", None)
    config = getattr(args, "prompt_data_config", None)
    if sum(bool(x) for x in (args.prompt_data, files, config)) != 1:
        raise ValueError("Select exactly one of --prompt-data, --prompt-data-files, --prompt-data-config")
    if config:
        value, directory = read_prompt_config(config)
        sources = []
        for entry in value.get("datasets", []):
            source = PromptSource(**entry)
            if not Path(source.path).is_absolute():
                source = PromptSource(**{**entry, "path": str(directory / source.path)})
            sources.append(source)
        size = getattr(args, "prompt_data_epoch_size", None)
        if size is None:
            size = value.get("epoch_size")
    elif files:
        ratios = getattr(args, "prompt_data_ratios", None)
        if ratios is not None and len(ratios) != len(files):
            raise ValueError("--prompt-data-ratios needs one ratio per --prompt-data-files entry")
        sources = [
            PromptSource(name=f"{i}_{Path(p).stem}", path=p, ratio=ratios[i] if ratios else None)
            for i, p in enumerate(files)
        ]
        size = getattr(args, "prompt_data_epoch_size", None)
    else:
        sources, size = [PromptSource(name=Path(args.prompt_data).stem, path=args.prompt_data)], None
    _validate_sources(sources)
    return sources, size


def _validate_sources(sources: list[PromptSource]) -> None:
    if not sources:
        raise ValueError("Prompt mixture needs nonempty, unique source names")
    for source in sources:
        if (
            not isinstance(source.name, str)
            or not source.name.strip()
            or not isinstance(source.path, str)
            or not source.path
        ):
            raise ValueError("Prompt source name and path must be nonempty strings")
        if source.category is not None and (not isinstance(source.category, str) or not source.category.strip()):
            raise ValueError("Prompt source category must be a nonempty string")
        if source.ratio is not None and (
            isinstance(source.ratio, bool)
            or not isinstance(source.ratio, (int, float))
            or not math.isfinite(source.ratio)
            or source.ratio < 0
        ):
            raise ValueError("Prompt source ratios must be finite and non-negative")
    if len({s.name for s in sources}) != len(sources):
        raise ValueError("Prompt source names must be unique")
    if any(s.ratio is not None for s in sources) and any(s.ratio is None for s in sources):
        raise ValueError("Specify ratios for every source, or omit all ratios for concatenation")
    if all(s.ratio == 0 for s in sources):
        raise ValueError("At least one prompt source ratio must be positive")


def prompt_fingerprint(args) -> str:
    sources, size = prompt_sources(args)
    inputs = []
    for source in sources:
        path = source.path.split("@[", 1)[0]
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        inputs.append({**vars(source), "sha256": digest.hexdigest()})
    settings = {
        key: getattr(args, key, None)
        for key in (
            "rollout_seed",
            "rollout_shuffle",
            "input_key",
            "label_key",
            "metadata_key",
            "tool_key",
            "rollout_max_prompt_len",
            "apply_chat_template",
            "apply_chat_template_kwargs",
            "chat_template_path",
            "hf_checkpoint",
            "n_samples_per_prompt",
            "multimodal_keys",
        )
    }
    value = {"sources": inputs, "epoch_size": size, "settings": settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
