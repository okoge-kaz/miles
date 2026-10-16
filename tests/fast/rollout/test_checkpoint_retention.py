from pathlib import Path

import pytest

from miles.rollout.checkpoint_retention import prune_rollout_checkpoints


def _save(root: Path, iteration: int, *, committed: bool = True) -> None:
    directory = root / "rollout"
    directory.mkdir(exist_ok=True)
    (directory / f"global_dataset_state_dict_{iteration}.pt").write_text("cursor")
    replay = directory / f"replay_{iteration:07d}"
    replay.mkdir(exist_ok=True)
    if committed:
        (replay / "COMMITTED.json").write_text("{}")
    (root / "hf" / str(iteration)).mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("retain_interval", [None, 50])
def test_retains_latest_and_completed_step_milestones(tmp_path, retain_interval):
    saved = [0, *range(9, 120, 10), 122]
    for iteration in saved:
        _save(tmp_path, iteration)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        prune_rollout_checkpoints(
            str(tmp_path), rollout_id=iteration, retain_interval=retain_interval, replay_required=True
        )
    expected = set(saved) if retain_interval is None else {49, 99, 122}
    directory = tmp_path / "rollout"
    assert {int(p.stem.removeprefix("global_dataset_state_dict_")) for p in directory.glob("*.pt")} == expected
    assert {int(p.name.removeprefix("replay_")) for p in directory.glob("replay_*")} == expected
    assert {int(p.name) for p in (tmp_path / "hf").iterdir()} == set(saved)


@pytest.mark.parametrize("tracker", [None, "release", "19", "29"])
def test_waits_for_complete_model_and_replay(tmp_path, tracker):
    _save(tmp_path, 9)
    _save(tmp_path, 19, committed=False)
    if tracker is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker)
    prune_rollout_checkpoints(str(tmp_path), rollout_id=19, retain_interval=50, replay_required=True)
    assert (tmp_path / "rollout/replay_0000009").is_dir()
    assert (tmp_path / "rollout/global_dataset_state_dict_9.pt").is_file()


def test_keeps_async_pending_models_and_future_or_unrecognized_sidecars(tmp_path):
    for iteration in (9, 19, 29, 39):
        _save(tmp_path, iteration)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("29")
    model = tmp_path / "iter_0000009"
    model.mkdir()
    directory = tmp_path / "rollout"
    (directory / "replay_0000001").mkdir()  # Uncommitted snapshot.
    (directory / "replay_0000002.tmp").mkdir()
    (directory / "global_dataset_state_dict_01.pt").write_text("unrecognized")
    (directory / "global_dataset_state_dict_3.pt").symlink_to(directory / "global_dataset_state_dict_9.pt")
    (directory / "replay_0000003").symlink_to(directory / "replay_0000009", target_is_directory=True)
    prune_rollout_checkpoints(str(tmp_path), rollout_id=39, retain_interval=50, replay_required=True)
    assert not (directory / "replay_0000019").exists()
    assert not (directory / "global_dataset_state_dict_19.pt").exists()
    for iteration in (9, 29, 39):
        assert (directory / f"replay_{iteration:07d}").is_dir()
        assert (directory / f"global_dataset_state_dict_{iteration}.pt").is_file()
    assert (directory / "replay_0000001").is_dir()
    assert (directory / "replay_0000002.tmp").is_dir()
    assert (directory / "global_dataset_state_dict_01.pt").is_file()
    assert (directory / "global_dataset_state_dict_3.pt").is_symlink()
    assert (directory / "replay_0000003").is_symlink()


def test_cursor_only_without_replay(tmp_path):
    directory = tmp_path / "rollout"
    directory.mkdir()
    for iteration in (9, 19):
        (directory / f"global_dataset_state_dict_{iteration}.pt").touch()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("19")
    prune_rollout_checkpoints(str(tmp_path), rollout_id=19, retain_interval=50, replay_required=False)
    assert not (directory / "global_dataset_state_dict_9.pt").exists()
    assert (directory / "global_dataset_state_dict_19.pt").exists()


def test_does_not_prune_through_rollout_directory_symlink(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "rollout").symlink_to(external, target_is_directory=True)
    for iteration in (9, 19):
        (external / f"global_dataset_state_dict_{iteration}.pt").touch()
    (root / "latest_checkpointed_iteration.txt").write_text("19")
    prune_rollout_checkpoints(str(root), rollout_id=19, retain_interval=50, replay_required=False)
    assert len(list(external.iterdir())) == 2
