import json
from pathlib import Path

from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX, resolve_file_arg


def read_prompt_config(value: str | Path) -> tuple[dict, Path | None]:
    value = str(value)
    inline = value.startswith(PSEUDO_FILE_PREFIX)
    text = resolve_file_arg(value)
    if inline or Path(value).suffix == ".json":
        config = json.loads(text)
    else:
        # Keep JSON-only job previews independent of optional YAML dependencies.
        import yaml

        config = yaml.safe_load(text)
    if not isinstance(config, dict) or set(config) - {"datasets", "epoch_size"}:
        raise ValueError("Prompt mixture config needs datasets and optional epoch_size")
    if not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("Prompt mixture config needs a nonempty datasets list")
    for source in config["datasets"]:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"]:
            raise ValueError("Prompt source needs a nonempty path")
        if inline and not Path(source["path"]).is_absolute():
            raise ValueError("Inline prompt config requires absolute source paths")
    return config, None if inline else Path(value).parent
