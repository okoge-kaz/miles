"""Opt-in assertions that the real trainer consumes every loaded R3 stream.

Use through --custom-megatron-init-path in the disposable four-node validation.
This adds device synchronization and file IO, so it is not a throughput probe.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from miles.utils.replay_base import Replay, routing_replay_manager


def install(args):
    assert args.use_rollout_routing_replay and args.use_replay_buffer
    assert args.recompute_granularity == "full"
    manager = routing_replay_manager
    output = Path(args.dump_details) / "replay_routing_probe"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{os.environ['SLURM_JOB_ID']}-rank{dist.get_rank()}.jsonl"

    def checked_pop(original, index_field):
        def pop(replay):
            expected = replay.top_indices_list[getattr(replay, index_field)]
            actual = original(replay)
            assert torch.equal(actual.cpu(), expected), "R3 routing changed between buffer and model forward"
            return actual

        return pop

    Replay.pop_forward = checked_pop(Replay.pop_forward, "forward_index")
    Replay.pop_backward = checked_pop(Replay.pop_backward, "backward_index")
    original_clear = manager.clear_all

    def checked_clear():
        assert manager.enabled and manager.replays, "No R3 streams were registered in the trainer"
        streams = []
        for replay in manager.replays:
            batches = len(replay.top_indices_list)
            assert batches > 0, "A trainer R3 stream received no rollout routing"
            assert replay.forward_index == batches, "Training forward did not consume every R3 microbatch"
            assert replay.backward_index == batches, "Backward recomputation did not consume every R3 microbatch"
            streams.append(
                {
                    "layer": replay.stream_idx,
                    "microbatches": batches,
                    "forward_consumed": replay.forward_index,
                    "backward_consumed": replay.backward_index,
                    "token_rows": sum(t.shape[0] for t in replay.top_indices_list),
                }
            )
        with path.open("a") as stream:
            stream.write(json.dumps({"rank": dist.get_rank(), "streams": streams}) + "\n")
        print(f"R3 consumption verified: rank={dist.get_rank()}, streams={len(streams)}, path={path}", flush=True)
        original_clear()

    manager.clear_all = checked_clear
