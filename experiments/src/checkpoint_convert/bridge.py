import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import dist_checkpointing, mpu
from megatron.core.enums import ModelType
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import miles_plugins.megatron_bridge  # noqa: F401 -- registers Miles' official bridges
from experiments.src.checkpoint_convert.assets import copy_hf_assets
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.megatron_bridge_utils import patch_megatron_model
from tools.convert_hf_to_torch_dist import get_args


def _save_release(args, model) -> None:
    save_checkpoint(1, model, None, None, 0)
    if dist.get_rank() == 0:
        source = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        destination = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source, destination)
    dist.barrier()
    if dist.get_rank() == 0:
        Path(get_checkpoint_tracker_filename(args.save)).write_text("release")


def _load_weights(model, checkpoint_dir: str) -> None:
    # Match Megatron generate_state_dict, including its data-parallel metadata.
    metadata = {"dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True)}
    keys = ["model"] if len(model) == 1 else [f"model{index}" for index in range(len(model))]
    sharded_state = {key: chunk.sharded_state_dict(metadata=metadata) for key, chunk in zip(keys, model, strict=True)}
    loaded = dist_checkpointing.load(sharded_state_dict=sharded_state, checkpoint_dir=checkpoint_dir)
    for key, chunk in zip(keys, model, strict=True):
        chunk.load_state_dict(loaded[key], strict=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("hf_mcore", "mcore_hf"):
        raise SystemExit("usage: bridge.py {hf_mcore|mcore_hf} [official Megatron conversion flags]")
    direction = sys.argv.pop(1)
    configure_logger_raw()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}"))
    try:
        args = get_args()
        # Miles normally derives this during training validation, which the
        # offline Megatron argument parser does not run.
        args.variable_seq_lengths = True
        if not any(argument.split("=", 1)[0] == "--moe-token-dispatcher-type" for argument in sys.argv[1:]):
            # Bridge's shared-expert overlap requires alltoall or flex. The
            # offline Megatron parser otherwise supplies its allgather default.
            args.moe_token_dispatcher_type = "alltoall"
        if args.megatron_to_hf_mode != "bridge":
            raise ValueError("this adapter requires --megatron-to-hf-mode bridge")
        # Conversion retains the HF draft head even when RL training disables it.
        os.environ["MILES_NEMOTRONH_KEEP_MTP"] = "1"
        init(args)
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        enforce_marked_param_dtypes(model)
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if direction == "hf_mcore":
            with patch_megatron_model(model):
                bridge.load_hf_weights(model)
            _save_release(args, model)
        else:
            if not args.load:
                raise ValueError("bridge export requires --load <torch_dist iteration directory>")
            _load_weights(model, args.load)
            with patch_megatron_model(model):
                bridge.save_hf_pretrained(model, path=args.save, source_path=args.hf_checkpoint)
            if dist.get_rank() == 0:
                copy_hf_assets(Path(args.hf_checkpoint), Path(args.save))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
