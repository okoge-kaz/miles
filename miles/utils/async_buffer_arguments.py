def add_buffer_arguments(parser) -> None:
    parser.add_argument(
        "--log-category-staleness",
        action="store_true",
        default=False,
        help="Actual selected-batch category fractions and measured version-lag distributions.",
    )
