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
        "--log-category-staleness",
        action="store_true",
        default=False,
        help="Actual selected-batch category fractions and measured version-lag distributions.",
    )


def validate_buffer_arguments(args) -> None:
    custom_ordering = (
        getattr(args, "async_data_buffer_order", "fifo") != "fifo"
        or getattr(args, "async_staleness_metric", "oldest") != "oldest"
        or getattr(args, "custom_staleness_function_path", None) is not None
    )
    if custom_ordering and not args.fully_async:
        raise ValueError("Async buffer ordering/staleness options require --fully-async")
    if custom_ordering and args.custom_async_data_buffer_path is not None:
        raise ValueError("Async buffer ordering/staleness options require the built-in DataBuffer")
