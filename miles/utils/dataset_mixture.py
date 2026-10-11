import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from miles.utils.dataset_mixture_config import read_prompt_config
from miles.utils.types import RewardSpec, Sample


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


def _validate_epoch_size(epoch_size: int) -> None:
    if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size < 1:
        raise ValueError("Prompt mixture epoch size must be a positive integer")


def source_quotas(weights: Sequence[float], epoch_size: int) -> list[int]:
    _validate_epoch_size(epoch_size)
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Prompt mixture ratios must be finite, non-negative, and have a positive sum")
    scale = max(weights)
    normalized = [w / scale for w in weights]
    total = sum(normalized)
    exact = [epoch_size * w / total for w in normalized]
    counts = [math.floor(x) for x in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: epoch_size - sum(counts)]:
        counts[i] += 1
    return counts


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
    if size is not None:
        _validate_epoch_size(size)
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


class MixtureDataset:
    def __init__(
        self, sources: list[PromptSource], datasets: list, *, seed: int, epoch_size: int | None, shuffle: bool
    ):
        _validate_sources(sources)
        if len(sources) != len(datasets):
            raise ValueError("Prompt source and dataset counts differ")
        self._sources = sources
        self._rows = [d.origin_samples for d in datasets]
        if any(not rows and source.ratio != 0 for source, rows in zip(sources, self._rows, strict=True)):
            raise ValueError("A nonzero prompt source is empty after filtering")
        self.seed = seed
        self._shuffle = shuffle
        size = epoch_size if epoch_size is not None else sum(map(len, self._rows))
        weights = [s.ratio for s in sources] if sources[0].ratio is not None else list(map(len, self._rows))
        self._quotas = source_quotas(weights, size)
        # Explicit source categories define a stable target even across async epoch boundaries.
        # Row-level categories can change with subsampling; do not invent targets for them.
        self.category_target_group_fractions = None
        if all(source.category is not None for source in sources):
            category_counts = Counter()
            for source, count in zip(sources, self._quotas, strict=True):
                category_counts[source.category.strip()] += count
            self.category_target_group_fractions = {name: count / size for name, count in category_counts.items()}
        for source, rows in zip(sources, self._rows, strict=True):
            for sample in rows:
                sample.metadata = {**(sample.metadata or {}), "dataset_source": source.name}
                if source.category is not None:
                    sample.metadata["category"] = source.category
                else:
                    sample.metadata.setdefault("category", source.name)
                if source.generate_function_path is not None:
                    sample.generate_function_path = source.generate_function_path
                if source.rm_type is not None or source.custom_rm_path is not None:
                    sample.reward_spec = RewardSpec(rm_type=source.rm_type, custom_rm_path=source.custom_rm_path)
        self.origin_samples = [sample for rows in self._rows for sample in rows]
        self.epoch_id = -1
        self.samples: list[Sample] = []
        self.shuffle(0)

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        rng = random.Random(self.seed + new_epoch_id)
        result = []
        for rows, count in zip(self._rows, self._quotas, strict=True):
            remaining = count
            while remaining:
                cycle = list(rows)
                if self._shuffle:
                    rng.shuffle(cycle)
                take = min(remaining, len(cycle))
                result.extend(cycle[:take])
                remaining -= take
        if self._shuffle:
            rng.shuffle(result)
        self.samples = result
        self.epoch_id = new_epoch_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def build_mixture_dataset(args, *, dataset_factory: Callable, dataset_kwargs: dict) -> MixtureDataset:
    sources, size = prompt_sources(args)
    datasets = []
    for source in sources:
        kwargs = dict(dataset_kwargs)
        for field, target in (
            ("input_key", "prompt_key"),
            ("label_key", "label_key"),
            ("metadata_key", "metadata_key"),
            ("tool_key", "tool_key"),
        ):
            if (value := getattr(source, field)) is not None:
                kwargs[target] = value
        datasets.append(dataset_factory(source.path, **kwargs))
    return MixtureDataset(sources, datasets, seed=args.rollout_seed, epoch_size=size, shuffle=args.rollout_shuffle)


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
