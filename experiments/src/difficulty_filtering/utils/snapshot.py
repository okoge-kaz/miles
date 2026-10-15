import argparse
import shutil
from pathlib import Path


def snapshot(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("miles", "experiments/src"):
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(root / "release-lock.json", destination / "release-lock.json")


def main():
    parser = argparse.ArgumentParser(description="Freeze the filtering and verifier sources before job submission")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    snapshot(Path(__file__).resolve().parents[4], args.destination)


if __name__ == "__main__":
    main()
