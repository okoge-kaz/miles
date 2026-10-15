import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.src.difficulty_filtering.core.config import FilterConfig
from experiments.src.difficulty_filtering.core.runtime import serve, server_command


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = FilterConfig(
            model_path="/model",
            prompt_data="/data",
            measurement_dir="/measurement",
            filtered_output="/output",
            policy="test",
        )

    def test_engines_have_disjoint_gpus_ports_and_seeds_without_internal_dp(self):
        for tp, count in [(1, 8), (2, 4)]:
            config = replace(self.config, tensor_parallel_size=tp, num_engines=count)
            ports, seeds, gpus = [], [], []
            for index in range(count):
                command = server_command(config, index)
                flags = dict(zip(command[3::2], command[4::2], strict=True))
                self.assertEqual(flags["--dp-size"], "1")
                self.assertEqual(int(flags["--tp-size"]), tp)
                ports.append(int(flags["--port"]))
                seeds.append(int(flags["--random-seed"]))
                first_gpu = int(flags["--base-gpu-id"])
                gpus.extend(range(first_gpu, first_gpu + tp))
            self.assertEqual(gpus, list(range(8)))
            self.assertEqual(seeds, list(range(1234, 1234 + count)))
            self.assertEqual(len(set(ports)), count)

    def test_partial_startup_failure_cleans_every_started_engine(self):
        first = Mock(pid=123, returncode=0)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("experiments.src.difficulty_filtering.core.runtime.socket.socket"),
            patch(
                "experiments.src.difficulty_filtering.core.runtime.subprocess.Popen", side_effect=[first, OSError("launch")]
            ),
            patch("experiments.src.difficulty_filtering.core.runtime.os.killpg") as kill,
            self.assertRaisesRegex(OSError, "launch"),
        ):
            with serve(self.config, Path(directory)):
                self.fail("Failed pool must never accept requests")
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], first.pid)
        first.wait.assert_called_once()

    def test_measurement_failure_cleans_entire_pool(self):
        processes = [Mock(pid=100 + index, returncode=0) for index in range(8)]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("experiments.src.difficulty_filtering.core.runtime.socket.socket"),
            patch("experiments.src.difficulty_filtering.core.runtime.subprocess.Popen", side_effect=processes),
            patch("experiments.src.difficulty_filtering.core.runtime.wait_for_servers", return_value=["http://test"] * 8),
            patch("experiments.src.difficulty_filtering.core.runtime.os.killpg") as kill,
            self.assertRaisesRegex(RuntimeError, "measurement"),
        ):
            with serve(self.config, Path(directory)):
                raise RuntimeError("measurement")
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {process.pid for process in processes})
        for process in processes:
            process.wait.assert_called_once()
