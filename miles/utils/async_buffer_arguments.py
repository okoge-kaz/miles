from miles.utils.dataset_mixture import prompt_sources


def add_buffer_arguments(parser) -> None:
    parser.add_argument(
        "--async-data-buffer-order",
        choices=("fifo", "oldest-policy-first"),
        default="fifo",
        help="Ready-group order. Oldest-policy-first uses the minimum observed generated version; unknown versions come last.",
    )
    parser.add_argument(
        "--async-staleness-metric",
        choices=("oldest", "newest", "token-mean", "submission", "first-prefill", "ready"),
        default="oldest",
        help="Definition used by --max-weight-staleness. Lifecycle references require measured provenance.",
    )
    parser.add_argument(
        "--custom-staleness-function-path",
        default=None,
        help="Optional function(args, DataBufferInput, current_version) -> lag or None.",
    )
    parser.add_argument(
        "--use-replay-data-buffer",
        "--use-replay-buffer",
        dest="use_replay_data_buffer",
        action="store_true",
        default=False,
        help="Persist fully-async ready groups and regenerate unfinished prompts on resume.",
    )
    parser.add_argument(
        "--replay-data-buffer-mode",
        choices=("rollout", "inflight"),
        default="rollout",
        help="rollout regenerates unfinished prompts; inflight saves and continues single-turn token prefixes.",
    )
    parser.add_argument(
        "--replay-data-buffer-async-restore",
        "--replay-buffer-async-restore",
        dest="replay_data_buffer_async_restore",
        action="store_true",
        default=False,
        help="Start training after the first loaded part; restore later parts in background with capacity backpressure.",
    )
    parser.add_argument(
        "--log-category-staleness",
        action="store_true",
        default=False,
        help="Actual selected-batch category fractions and measured version-lag distributions.",
    )
    parser.add_argument(
        "--log-policy-lag-metrics",
        action="store_true",
        default=False,
        help="Signed current-training-forward logprob minus rollout logprob, pooled globally and per category.",
    )


def validate_buffer_arguments(args) -> None:
    replay = getattr(args, "use_replay_data_buffer", False)
    custom_ordering = (
        getattr(args, "async_data_buffer_order", "fifo") != "fifo"
        or getattr(args, "async_staleness_metric", "oldest") != "oldest"
        or getattr(args, "custom_staleness_function_path", None) is not None
    )
    if custom_ordering and not args.fully_async:
        raise ValueError("Async buffer ordering/staleness options require --fully-async")
    if custom_ordering and args.custom_async_data_buffer_path is not None:
        raise ValueError("Async buffer ordering/staleness options require the built-in DataBuffer")
    if getattr(args, "replay_data_buffer_async_restore", False) and not replay:
        raise ValueError("--replay-data-buffer-async-restore requires --use-replay-data-buffer")
    mode = getattr(args, "replay_data_buffer_mode", "rollout")
    if mode == "inflight":
        if not replay:
            raise ValueError("--replay-data-buffer-mode inflight requires --use-replay-data-buffer")
        standard_generators = (None, "miles.rollout.generate_hub.single_turn.generate")
        if args.custom_generate_function_path not in standard_generators or getattr(args, "use_session_server", False):
            raise ValueError("Inflight ReplayDataBuffer requires standard single-turn generation")
        if any(getattr(args, key, False) for key in ("use_rollout_routing_replay", "use_rollout_indexer_replay")):
            raise ValueError("Inflight ReplayDataBuffer does not support routing/indexer replay")
        if getattr(args, "prompt_data_files", None) or getattr(args, "prompt_data_config", None):
            if any(source.generate_function_path not in standard_generators for source in prompt_sources(args)[0]):
                raise ValueError("Inflight ReplayDataBuffer does not support per-source custom generators")
    if replay:
        if not args.fully_async or not args.rollout_global_dataset:
            raise ValueError("ReplayDataBuffer requires --fully-async and the global prompt dataset")
        if args.save is None or args.save_interval is None:
            raise ValueError("ReplayDataBuffer requires --save and --save-interval")
        if args.custom_async_data_buffer_path is not None:
            raise ValueError("ReplayDataBuffer persistence requires the built-in DataBuffer snapshot contract")
        if args.train_backend != "megatron" or args.indep_dp or args.use_critic or args.multi_lora:
            raise ValueError(
                "ReplayDataBuffer currently supports a single Megatron actor policy without independent-DP recovery/critic/multi-LoRA"
            )
        if args.debug_train_only or args.debug_rollout_only or args.load_debug_rollout_data:
            raise ValueError("ReplayDataBuffer requires actual coupled training and rollout, not debug injection")
        if getattr(args, "replay_data_buffer_async_restore", False) and args.async_data_buffer_capacity_factor < 1:
            raise ValueError("Async ReplayDataBuffer restore requires capacity for at least one batch")
