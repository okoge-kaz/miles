import importlib.util
import json
import signal
import sys
from pathlib import Path


def main():
    grader_path, request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    spec = importlib.util.spec_from_file_location("nemo_lcb_grader", grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    (result_path.parent / "started").touch()
    try:
        results, metadata = grader.run_test(request["tests"], test=request["code"], timeout=request["timeout"])
        result = {"results": results, "metadata": metadata or {}}
    except grader.TimeoutException:
        result = {"failure_kind": "execution_time_limit"}
    except Exception:
        result = {"failure_kind": "grader_exception"}
    finally:
        signal.alarm(0)
    result_path.write_text(json.dumps(result, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
