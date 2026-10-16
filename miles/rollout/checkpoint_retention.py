import logging
import re
import shutil
from pathlib import Path

from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration

logger = logging.getLogger(__name__)


def prune_rollout_checkpoints(
    save: str | None, *, rollout_id: int, retain_interval: int | None, replay_required: bool
) -> None:
    if save is None or retain_interval is None:
        return
    if retain_interval <= 0:
        raise ValueError("Checkpoint retention interval must be positive")
    root = Path(save)
    directory = root / "rollout"
    if root.is_symlink() or directory.is_symlink() or not directory.is_dir():
        return
    try:
        committed = read_checkpoint_tracker_iteration(root)
        if committed is None or committed > rollout_id:
            return
        # Async model saves can lag the cursor. Keep everything newer than the
        # completed model and require its sidecars before removing older ones.
        if not (directory / f"global_dataset_state_dict_{committed}.pt").is_file():
            return
        if replay_required and not (directory / f"replay_{committed:07d}" / "COMMITTED.json").is_file():
            return
        for path in directory.iterdir():
            _prune_old_checkpoint(path, root=root, committed=committed, retain_interval=retain_interval)
    except OSError:
        # Retention failure must not invalidate the checkpoint that was just saved.
        logger.warning("Could not prune rollout checkpoints in %s", directory, exc_info=True)


def _prune_old_checkpoint(path: Path, *, root: Path, committed: int, retain_interval: int) -> None:
    cursor = re.fullmatch(r"global_dataset_state_dict_(0|[1-9][0-9]*)\.pt", path.name)
    replay = re.fullmatch(r"replay_([0-9]{7,})", path.name)
    if path.is_symlink() or (match := cursor or replay) is None:
        return
    iteration = int(match[1])
    if iteration >= committed or (iteration + 1) % retain_interval == 0:
        return
    model = root / f"iter_{iteration:07d}"
    # Megatron owns model deletion. Keep sidecars while its deletion is pending,
    # or if a different image/policy deliberately keeps that model generation.
    if model.exists() or model.is_symlink():
        return
    if cursor and path.is_file():
        path.unlink()
    elif replay and path.is_dir() and (path / "COMMITTED.json").is_file():
        shutil.rmtree(path)
    else:
        return
    logger.info("Removed expired rollout checkpoint %s", path)
