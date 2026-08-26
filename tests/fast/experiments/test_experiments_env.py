from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def test_experiments_env_does_not_read_repository_dotenv(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[3] / "experiments" / "env.sh"
    checkout = tmp_path / "checkout"
    script = checkout / "experiments" / "env.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(source, script)
    (checkout / ".env").write_text("MILES_DOTENV_SENTINEL=must-not-load\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "MILES_REPO": str(checkout),
            "SHARED_WS": str(tmp_path / "shared"),
            "WS": str(tmp_path / "workspace"),
            "WANDB_MODE": "disabled",
        }
    )
    environment.pop("MILES_DOTENV_SENTINEL", None)

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; [[ -z "${MILES_DOTENV_SENTINEL:-}" ]]', "bash", str(script)],
        check=False,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
