import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args()
    root = Path("/opt/miles/environments")
    provenance = json.loads((root / "provenance.json").read_text())
    for command in provenance["sandbox_commands"]:
        if shutil.which(command) is None:
            raise RuntimeError(f"Missing sandbox command: {command}")
    if args.sandbox:
        import tempfile

        from experiments.src.nemo_gym.sandbox.execution import execute

        with tempfile.TemporaryDirectory() as work:
            status, output, _ = execute(
                ["/bin/echo", "sandbox-ready"], writable=[work], timeout=10, timeout_kind="sandbox_probe_timeout"
            )
            if status != 0 or output.strip() != "sandbox-ready":
                raise RuntimeError("CPU sandbox could not execute its readiness probe")
    print(json.dumps({"sandbox_probe": args.sandbox, "status": "ok"}), flush=True)


if __name__ == "__main__":
    main()
