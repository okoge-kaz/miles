import base64
import json
import random
from argparse import Namespace
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.utils.dataset_mixture import MixtureDataset, PromptSource, prompt_fingerprint, prompt_sources, source_quotas
from miles.utils.types import Sample


def test_mixture_exact_quotas_shuffle_resume_and_rng_isolation():
    sources = [
        PromptSource("math", "math.jsonl", 0.75, rm_type="deepscaler"),
        PromptSource(
            "code", "code.jsonl", 0.25, generate_function_path="example.generate", custom_rm_path="example.reward"
        ),
    ]
    datasets = [
        SimpleNamespace(origin_samples=[Sample(prompt=f"m{i}") for i in range(2)]),
        SimpleNamespace(origin_samples=[Sample(prompt=f"c{i}") for i in range(3)]),
    ]
    state = random.getstate()
    mixed = MixtureDataset(sources, datasets, seed=17, epoch_size=20, shuffle=True)
    assert random.getstate() == state
    assert Counter(s.metadata["category"] for s in mixed.samples) == {"math": 15, "code": 5}
    first = [s.prompt for s in mixed.samples]
    mixed.shuffle(3)
    third = [s.prompt for s in mixed.samples]
    assert third != first
    mixed.shuffle(0)
    assert [s.prompt for s in mixed.samples] == first
    mixed.shuffle(3)
    assert [s.prompt for s in mixed.samples] == third
    code = next(s for s in mixed.samples if s.metadata["category"] == "code")
    assert code.generate_function_path == "example.generate"
    assert code.reward_spec.custom_rm_path == "example.reward"


def test_unweighted_concatenation_and_zero_ratio():
    datasets = [
        SimpleNamespace(origin_samples=[Sample(prompt="a")]),
        SimpleNamespace(origin_samples=[Sample(prompt="b"), Sample(prompt="c")]),
    ]
    mixed = MixtureDataset(
        [PromptSource("a", "a"), PromptSource("b", "b")], datasets, seed=1, epoch_size=None, shuffle=False
    )
    assert [s.prompt for s in mixed.samples] == ["a", "b", "c"]
    assert source_quotas([1, 0, 2], 10) == [3, 0, 7]


def test_empty_source_is_allowed_only_when_its_ratio_is_zero():
    sources = [PromptSource("math", "math", 1), PromptSource("code", "code", 0)]
    datasets = [SimpleNamespace(origin_samples=[Sample(prompt="m")]), SimpleNamespace(origin_samples=[])]
    mixed = MixtureDataset(sources, datasets, seed=1, epoch_size=3, shuffle=True)
    assert [sample.prompt for sample in mixed.samples] == ["m", "m", "m"]
    sources[1] = PromptSource("code", "code", 1)
    with pytest.raises(ValueError, match="empty after filtering"):
        MixtureDataset(sources, datasets, seed=1, epoch_size=3, shuffle=True)


@pytest.mark.parametrize("size", [0, True, 1.5])
def test_config_epoch_size_is_validated_before_loading_data(tmp_path, size):
    config = tmp_path / "mix.json"
    config.write_text(json.dumps({"datasets": [{"name": "math", "path": "missing.jsonl"}], "epoch_size": size}))
    args = Namespace(prompt_data=None, prompt_data_config=str(config))
    with pytest.raises(ValueError, match="positive integer"):
        prompt_sources(args)


@pytest.mark.parametrize("weights,size", [([0, 0], 3), ([float("nan"), 1], 3), ([-1, 2], 3), ([1], 0), ([1], True)])
def test_bad_ratios_rejected(weights, size):
    with pytest.raises(ValueError):
        source_quotas(weights, size)


def test_config_relative_paths_hooks_and_fingerprint_change(tmp_path):
    data = tmp_path / "math.jsonl"
    data.write_text('{"prompt":"x"}\n')
    config = tmp_path / "mix.yaml"
    config.write_text(
        "datasets:\n  - name: math\n    path: math.jsonl\n    ratio: 1\n    category: math/easy\nepoch_size: 8\n"
    )
    args = Namespace(prompt_data=None, prompt_data_config=str(config), rollout_seed=42)
    sources, size = prompt_sources(args)
    assert sources[0].path == str(data) and sources[0].category == "math/easy" and size == 8
    digest = prompt_fingerprint(args)
    data.write_text('{"prompt":"y"}\n')
    assert digest != prompt_fingerprint(args)
    args.prompt_data = str(data)
    with pytest.raises(ValueError, match="exactly one"):
        prompt_sources(args)


def test_category_targets_use_rounded_quotas_and_merge_sources():
    sources = [
        PromptSource("m1", "m1", 1, category="math"),
        PromptSource("m2", "m2", 1, category="math"),
        PromptSource("code", "code", 1, category="code"),
        PromptSource("mcqa", "mcqa", 0, category="mcqa"),
    ]
    datasets = [SimpleNamespace(origin_samples=[Sample(prompt=str(i))]) for i in range(4)]
    mixed = MixtureDataset(sources, datasets, seed=1, epoch_size=5, shuffle=True)
    target = mixed.category_target_group_fractions
    assert target == {"math": 0.8, "code": 0.2, "mcqa": 0}
    mixed.shuffle(3)
    assert mixed.category_target_group_fractions == target
    sources[0] = PromptSource("m1", "m1", 1)
    mixed = MixtureDataset(sources, datasets, seed=1, epoch_size=5, shuffle=True)
    assert mixed.category_target_group_fractions is None


def test_inline_config_preserves_ratios_hooks_and_rejects_relative_paths(tmp_path):
    config = {
        "datasets": [
            {"name": "qa", "path": str(tmp_path / "qa.jsonl"), "ratio": 2, "custom_rm_path": "qa.reward"},
            {"name": "tool", "path": str(tmp_path / "tool.jsonl"), "ratio": 3, "generate_function_path": "tool.run"},
        ],
        "epoch_size": 50,
    }

    def arguments():
        payload = "base64:" + base64.b64encode(json.dumps(config).encode()).decode()
        return Namespace(prompt_data=None, prompt_data_config=payload)

    sources, size = prompt_sources(arguments())
    assert size == 50 and source_quotas([source.ratio for source in sources], size) == [20, 30]
    assert sources[0].custom_rm_path == "qa.reward" and sources[1].generate_function_path == "tool.run"
    config["datasets"][0]["path"] = "qa.jsonl"
    with pytest.raises(ValueError, match="absolute source paths"):
        prompt_sources(arguments())
