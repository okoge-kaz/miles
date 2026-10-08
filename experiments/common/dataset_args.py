import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path


def prompt_arguments(environment: Mapping[str, str]) -> list[str]:
    config = environment.get("PROMPT_DATA_CONFIG", "")
    files = environment.get("PROMPT_DATA_FILES", "")
    if config and files:
        raise ValueError("Choose PROMPT_DATA_CONFIG or PROMPT_DATA_FILES")
    if config:
        arguments = ["--prompt-data-config", config]
    elif files:
        paths = json.loads(files)
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError("PROMPT_DATA_FILES must be a nonempty JSON array of paths")
        arguments = ["--prompt-data-files", *paths]
    else:
        arguments = ["--prompt-data", environment["PROMPT_DATA"]]
    ratios = environment.get("PROMPT_DATA_RATIOS", "")
    if ratios:
        values = json.loads(ratios)
        if not files or not isinstance(values, list) or len(values) != len(paths):
            raise ValueError("PROMPT_DATA_RATIOS must be a JSON array matching PROMPT_DATA_FILES")
        if (
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)
            or sum(values) <= 0
        ):
            raise ValueError("PROMPT_DATA_RATIOS must be finite, non-negative and have a positive sum")
        arguments += ["--prompt-data-ratios", *map(str, values)]
    size = environment.get("PROMPT_DATA_EPOCH_SIZE", "")
    if size:
        if not (config or files) or int(size) < 1:
            raise ValueError("PROMPT_DATA_EPOCH_SIZE needs mixed sources and a positive integer")
        arguments += ["--prompt-data-epoch-size", size]
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve single or mixed prompt data settings")
    parser.add_argument("--json-array", choices=("PROMPT_DATA_FILES", "PROMPT_DATA_RATIOS"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--repo-root")
    args = parser.parse_args()
    try:
        arguments = prompt_arguments(os.environ)
        if args.json_array:
            arguments = list(map(str, json.loads(os.environ[args.json_array])))
        if args.check:
            check_prompt_paths(os.environ, data_root=Path(args.data_root), repo_root=Path(args.repo_root))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if not args.check:
        sys.stdout.buffer.write("\0".join(arguments).encode() + b"\0")


def check_prompt_paths(environment: Mapping[str, str], *, data_root: Path, repo_root: Path) -> None:
    def mounted_path(value: str) -> Path:
        real = value.split("@[", 1)[0]
        if real.startswith("/data/"):
            path = data_root / real.removeprefix("/data/")
        elif real.startswith("/root/miles/"):
            path = repo_root / real.removeprefix("/root/miles/")
        else:
            raise ValueError("Prompt paths must use the /data or /root/miles container mounts")
        if not path.is_file():
            raise ValueError(f"Missing prompt source/config: {path}")
        return path

    if environment.get("PROMPT_DATA_CONFIG"):
        path = mounted_path(environment["PROMPT_DATA_CONFIG"])
        if path.suffix == ".json":
            value = json.loads(path.read_text())
        else:
            # Optional only for a YAML config; ordinary previews stay stdlib-only.
            import yaml

            value = yaml.safe_load(path.read_text())
        for source in value["datasets"]:
            data = source["path"]
            if Path(data).is_absolute():
                mounted_path(data)
            elif not (path.parent / data.split("@[", 1)[0]).is_file():
                raise ValueError(f"Missing relative prompt source: {data}")
    else:
        files = environment.get("PROMPT_DATA_FILES")
        for path in json.loads(files) if files else [environment["PROMPT_DATA"]]:
            mounted_path(path)


if __name__ == "__main__":
    main()
