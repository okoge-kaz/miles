from pathlib import Path
from types import SimpleNamespace

from miles.utils.arguments import _resolve_eval_datasets


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_config(name: str):
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments" / "configs" / name),
        eval_prompt_data=None,
    )
    return _resolve_eval_datasets(args)


def test_aime_eval_config_uses_training_aligned_sampling_defaults():
    datasets = _load_config("eval_aime.yaml")

    assert [dataset.name for dataset in datasets] == ["aime24", "aime25", "aime26"]
    assert all(dataset.n_samples_per_eval_prompt == 16 for dataset in datasets)
    assert all(dataset.max_response_len == 16384 for dataset in datasets)


def test_gpqa_eval_config_routes_every_split_to_gpqa_reward():
    datasets = _load_config("eval_gpqa.yaml")

    assert [dataset.name for dataset in datasets] == ["gpqa_diamond", "gpqa_main", "gpqa_extended"]
    assert [dataset.n_samples_per_eval_prompt for dataset in datasets] == [8, 4, 4]
    assert all(dataset.max_response_len == 16384 for dataset in datasets)
    assert all(dataset.rm_type == "gpqa" for dataset in datasets)
