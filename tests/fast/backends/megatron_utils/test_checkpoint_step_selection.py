"""Check explicit archive selection against the installed Megatron path resolver."""

from argparse import Namespace
from pathlib import Path

import pytest
from megatron.training.checkpointing import get_load_checkpoint_path_by_args

import miles.backends.megatron_utils.checkpoint as checkpoint


@pytest.fixture
def checkpoint_args(tmp_path, monkeypatch):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3\n")
    args = Namespace(load=str(tmp_path), ckpt_step=None)
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "is_lora_enabled", lambda _: False)
    previous_step = checkpoint.get_loaded_iteration()
    yield args
    checkpoint.set_loaded_iteration(previous_step)


@pytest.mark.parametrize(("selected", "expected"), [(None, 3), (0, 0), (2, 2)])
def test_explicit_checkpoint_step_selects_requested_archive(checkpoint_args, monkeypatch, selected, expected):
    checkpoint_args.ckpt_step = selected

    def load(**kwargs):
        resolved = Path(get_load_checkpoint_path_by_args(checkpoint_args))
        iteration = int(resolved.name.removeprefix("iter_"))
        checkpoint.set_loaded_iteration(checkpoint_args.ckpt_step if selected is not None else iteration)
        return iteration, 0

    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", load)
    iteration, _ = checkpoint.load_checkpoint([], None, None, {}, False)

    assert iteration == expected
    assert checkpoint_args.ckpt_step is selected
    assert type(checkpoint.get_loaded_iteration()) is int
    assert checkpoint.get_loaded_iteration() == expected


def test_zero_checkpoint_step_restored_when_loading_fails(checkpoint_args, monkeypatch):
    checkpoint_args.ckpt_step = 0

    def fail(**kwargs):
        checkpoint.set_loaded_iteration(checkpoint_args.ckpt_step)
        raise RuntimeError("checkpoint read failed")

    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", fail)
    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        checkpoint.load_checkpoint([], None, None, {}, False)

    assert type(checkpoint_args.ckpt_step) is int
    assert checkpoint_args.ckpt_step == 0
    assert type(checkpoint.get_loaded_iteration()) is int
