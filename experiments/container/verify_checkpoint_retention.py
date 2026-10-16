import argparse
import ast
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


class _ImmediateTask:
    def __init__(self, *, target, args, **kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


def _delete_checkpoint(save, iteration, *args):
    shutil.rmtree(Path(save) / f"iter_{iteration:07d}")


def _checkpoint_name(save, iteration, **kwargs):
    return str(Path(save) / f"iter_{iteration:07d}")


def _finalizer(source, directory, *, retain_interval, asynchronous):
    # Exercise the installed completion callback without importing CUDA on a CPU build node.
    module = ast.parse(source.read_text())
    callback = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "iter_finalize_fn"
        and "save_retain_interval" in ast.unparse(node)
    )
    namespace = {
        "args": SimpleNamespace(
            save=str(directory),
            save_retain_interval=retain_interval,
            async_save=asynchronous,
            log_progress=False,
            async_ckpt_cpu_priority=None,
            async_ckpt_io_priority=None,
        ),
        "tracker_filename": str(directory / "latest_checkpointed_iteration.txt"),
        "release": False,
        "os": os,
        "open_file": open,
        "datetime": datetime,
        "print_rank_0": lambda *args: None,
        "tensor_mp_rank": 1,
        "tp_size_to_print": 1,
        "pipeline_mp_rank": 1,
        "pp_size_to_print": 1,
        "get_checkpoint_name": _checkpoint_name,
        "_async_delete_checkpoint_impl": _delete_checkpoint,
        "threading": SimpleNamespace(Thread=_ImmediateTask),
        "multiprocessing": SimpleNamespace(get_context=lambda _: SimpleNamespace(Process=_ImmediateTask)),
        "finalize_deletion_processes": lambda **kwargs: None,
        "_deletion_processes": [],
    }
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def verify_checkpoint_retention(source: Path):
    for asynchronous in (False, True):
        for interval in (None, 50):
            with tempfile.TemporaryDirectory(prefix="miles-checkpoint-retention-") as temporary:
                directory = Path(temporary)
                state = _finalizer(source, directory, retain_interval=interval, asynchronous=asynchronous)
                saved = []
                # Includes a first-step save, periodic saves, a non-periodic final save, and overwrite.
                for iteration in [0, *range(9, 120, 10), 122, 122]:
                    (directory / f"iter_{iteration:07d}").mkdir(exist_ok=True)
                    (directory / "hf" / str(iteration)).mkdir(parents=True, exist_ok=True)
                    (directory / "rollout" / f"replay_{iteration:07d}").mkdir(parents=True, exist_ok=True)
                    state["iteration"] = iteration
                    state["iter_finalize_fn"]()
                    saved.append(iteration)
                    expected = (
                        set(saved) if interval is None else {i for i in saved if (i + 1) % interval == 0} | {iteration}
                    )
                    actual = {int(path.name.removeprefix("iter_")) for path in directory.glob("iter_*")}
                    assert actual == expected, (asynchronous, interval, iteration, actual, expected)
                    assert (directory / "latest_checkpointed_iteration.txt").read_text() == str(iteration)
                    assert len(list((directory / "hf").iterdir())) == len(set(saved))
                    assert len(list((directory / "rollout").iterdir())) == len(set(saved))
                # Megatron's existing symlink protection must remain intact.
                protected = directory / "protected"
                protected.mkdir()
                shutil.rmtree(directory / "iter_0000122")
                (directory / "iter_0000122").symlink_to(protected, target_is_directory=True)
                (directory / "iter_0000129").mkdir()
                state["iteration"] = 129
                state["iter_finalize_fn"]()
                assert protected.is_dir() and (directory / "iter_0000122").is_symlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    verify_checkpoint_retention(parser.parse_args().source)
    print("Checkpoint retention verified: latest + completed-rollout milestones, sync and async callbacks")
