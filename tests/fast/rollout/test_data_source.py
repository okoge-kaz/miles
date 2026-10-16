import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import miles.rollout.data_source as source_module
from miles.rollout.data_source import RolloutDataSource


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    source._fingerprint = None
    source._is_mixture = False
    return source


def test_load_says_so_when_it_finds_no_state(tmp_path: Path, caplog) -> None:
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no dataset state under" in caplog.text


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no --load" in caplog.text


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    import torch

    from miles.rollout.data_source import compute_global_dataset_state_path

    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    source.load(rollout_id=3)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


def test_replay_cursor_resume_rejects_changed_prompt_content(tmp_path, monkeypatch):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": f"p{i}", "label": "answer"}) for i in range(4)) + "\n")
    monkeypatch.setattr(source_module, "load_tokenizer", lambda *a, **kw: object())
    monkeypatch.setattr(source_module, "load_processor", lambda *a, **kw: None)
    args = _make_args(
        rollout_global_dataset=True,
        use_replay_data_buffer=True,
        prompt_data=str(path),
        hf_checkpoint="model",
        chat_template_path=None,
        dump_details=None,
        rollout_max_prompt_len=None,
        input_key="prompt",
        label_key="label",
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=42,
        rollout_shuffle=True,
        n_samples_per_prompt=2,
    )
    source = RolloutDataSource(args)
    source.get_samples(3)
    state = source.state_dict()
    expected = [[(s.prompt, s.index, s.group_index) for s in group] for group in source.get_samples(3)]
    resumed = RolloutDataSource(args)
    resumed.load_state_dict(state)
    actual = [[(s.prompt, s.index, s.group_index) for s in group] for group in resumed.get_samples(3)]
    assert actual == expected
    path.write_text('{"prompt":"changed","label":"answer"}\n')
    with pytest.raises(ValueError, match="differ"):
        RolloutDataSource(args).load_state_dict(state)


@pytest.mark.parametrize("shuffle", [False, True])
def test_mixture_cursor_resume_across_multiple_epochs_and_source_change_rejection(tmp_path, monkeypatch, shuffle):
    paths = []
    for name in ("math", "code"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps({"prompt": f"{name}{i}", "label": "answer"}) for i in range(2)) + "\n")
        paths.append(str(path))
    monkeypatch.setattr(source_module, "load_tokenizer", lambda *a, **kw: object())
    monkeypatch.setattr(source_module, "load_processor", lambda *a, **kw: None)
    args = _make_args(
        rollout_global_dataset=True,
        prompt_data=None,
        prompt_data_files=paths,
        prompt_data_ratios=[0.75, 0.25],
        prompt_data_epoch_size=4,
        hf_checkpoint="model",
        chat_template_path=None,
        dump_details=None,
        rollout_max_prompt_len=None,
        input_key="prompt",
        label_key="label",
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=42,
        rollout_shuffle=shuffle,
        n_samples_per_prompt=2,
        save=str(tmp_path),
        load=str(tmp_path),
    )
    source = RolloutDataSource(args)
    source.get_samples(5)
    source.save(3)
    expected = [[(s.prompt, s.index, s.group_index, s.metadata) for s in g] for g in source.get_samples(7)]
    resumed = RolloutDataSource(args)
    resumed.load(3)
    actual = [[(s.prompt, s.index, s.group_index, s.metadata) for s in g] for g in resumed.get_samples(7)]
    assert actual == expected
    Path(paths[0]).write_text('{"prompt":"different","label":"answer"}\n')
    changed = RolloutDataSource(args)
    with pytest.raises(ValueError, match="differ"):
        changed.load(3)
