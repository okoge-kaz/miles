import json
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
