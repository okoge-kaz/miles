import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Publish a native Gym model config after SGLang reports the model ready"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new simulator run name: {args.output}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(args.base_url.rstrip("/") + "/models", timeout=5) as response:
                data = json.load(response)
            if args.model not in {item["id"] for item in data["data"]}:
                raise ValueError("SGLang served a different model identity")
            config = {"openai_base_url": args.base_url, "openai_model": args.model}
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(config, indent=2) + "\n")
            temporary.replace(args.output)
            args.output.with_suffix(".provenance.json").write_text(
                json.dumps({**config, "model_revision": args.revision}, indent=2) + "\n"
            )
            print(f"Ready: {args.output}", flush=True)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError("SGLang user simulator did not become ready")


if __name__ == "__main__":
    main()
