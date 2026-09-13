"""Download AIME-2024 evaluation data with a pinned revision.

Args:
    --dataset-root: Parent directory for the aime-2024 dataset directory.

Example:
    python -m scripts.download.datasets.eval.aime_2024 --dataset-root /data
"""

import argparse
from pathlib import Path

from scripts.download.datasets.common import stage_dataset

DATASET_NAME = "aime-2024"
DATASET_REVISION = "1c625e328db94ec7ef7ff169016b097c468d60b9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    stage_dataset(root=args.dataset_root, name=DATASET_NAME, revision=DATASET_REVISION)


if __name__ == "__main__":
    main()
