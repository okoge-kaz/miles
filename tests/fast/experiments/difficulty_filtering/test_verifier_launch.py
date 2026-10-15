import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("scenario,expected", [("success", 0), ("measure_failure", 7), ("gym_failure", 1)])
def test_native_filter_supervises_cpu_verifier_and_gpu_measurement(tmp_path, scenario, expected):
    run = tmp_path / "experiments/outputs/difficulty_filtering/test"
    (run / "source/experiments/src/difficulty_filtering").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"reward_backend": "code_gen", "startup_timeout": 5}))
    (tmp_path / "experiments/env.sh").write_text("""export MILES_JOB_NUM_NODES=1
export MILES_JOB_ID_SHORT=test
export CONTAINER_IMAGE=unused
export CONTAINER_MOUNTS=unused
_miles_build_container_command() { MILES_CONTAINER_COMMAND=(python3 "${MILES_REPO}/child.py" "$@"); }
""")
    (tmp_path / "child.py").write_text("""import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["MILES_REPO"])
args = sys.argv[1:]
is_gym = "experiments.src.difficulty_filtering.core.gym" in args
kind = "gym" if is_gym else "measure"
(root / (kind + "-args.json")).write_text(json.dumps(args))
state = root / "experiments/outputs/difficulty_filtering/test/gym-test-0"
endpoint = state / "endpoints.json"
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
try:
    if is_gym:
        endpoint.write_text("{}")
        while True:
            if os.environ["SCENARIO"] == "gym_failure" and (root / "measure-started").exists():
                sys.exit(23)
            time.sleep(0.05)
    else:
        assert endpoint.exists()
        (root / "measure-started").touch()
        if os.environ["SCENARIO"] == "success": sys.exit(0)
        if os.environ["SCENARIO"] == "measure_failure": sys.exit(7)
        while True: time.sleep(0.05)
finally:
    if is_gym: endpoint.unlink(missing_ok=True)
    (root / (kind + "-stopped")).touch()
""")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "experiments/scripts/difficulty_filtering/run.sh"),
            "/root/miles/experiments/outputs/difficulty_filtering/test/config.json",
            "smoke",
            "0",
            "32",
        ],
        env={"PATH": os.environ["PATH"], "MILES_REPO": str(tmp_path), "PBS_JOBID": "test", "SCENARIO": scenario},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    gym = json.loads((tmp_path / "gym-args.json").read_text())
    measure = json.loads((tmp_path / "measure-args.json").read_text())
    assert "--no-nv" in gym and "--nv" not in gym
    assert "--nv" in measure and "--no-nv" not in measure
    assert any(arg.startswith("NEMO_GYM_ENDPOINTS=") for arg in measure)
    assert (tmp_path / "gym-stopped").exists()
    assert (tmp_path / "measure-stopped").exists()
    assert not (run / "gym-test-0/endpoints.json").exists()


def test_external_verifier_is_reused_without_starting_a_child(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.src.difficulty_filtering.core.gym import serve_verifier

    endpoint = tmp_path / "endpoints.json"
    endpoint.write_text(json.dumps({"schema": 1, "services": {"code_gen": "http://127.0.0.1:30191"}}))
    monkeypatch.setenv("NEMO_GYM_ENDPOINTS", str(endpoint))
    with serve_verifier(SimpleNamespace(reward_backend="code_gen"), tmp_path):
        assert endpoint.is_file()
    assert endpoint.is_file()
    with pytest.raises(ValueError, match="no mcqa service"):
        with serve_verifier(SimpleNamespace(reward_backend="mcqa"), tmp_path):
            pytest.fail("Missing backend must be rejected")
