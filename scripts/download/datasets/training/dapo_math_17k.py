"""Download unfiltered DAPO-MATH-17K training data with a pinned revision.

Args:
    --dataset-root: Parent directory for the dapo-math-17k dataset directory.

Example:
    python -m scripts.download.datasets.training.dapo_math_17k --dataset-root /data
"""

import argparse
from pathlib import Path

from scripts.download.datasets.common import stage_dataset

DATASET_NAME = "dapo-math-17k"
DATASET_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    stage_dataset(root=args.dataset_root, name=DATASET_NAME, revision=DATASET_REVISION)


if __name__ == "__main__":
    main()
