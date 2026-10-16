import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path("/opt/miles/environments")
    root.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path(__file__).with_name("code_sandbox.requirements.json").read_text())
    for name in lock["sandbox_commands"]:
        if shutil.which(name) is None:
            raise RuntimeError(f"Missing sandbox command: {name}")
    (root / "provenance.json").write_text(json.dumps(lock, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify_code_sandbox.py"))], check=True)


if __name__ == "__main__":
    main()
