"""CPU-only checks for the September checkpoint figure refresh."""

import importlib.util
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments/outputs/reasoning_eval/wandb-collapse-20260901/plot_cross_cohort.py"
)
SPEC = importlib.util.spec_from_file_location("cross_cohort_refresh_test", MODULE_PATH)
CROSS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CROSS
SPEC.loader.exec_module(CROSS)


class FigureRefreshTests(unittest.TestCase):
    def test_reruns_are_not_spliced_into_historical_sources(self):
        historical = {source.namespace for source in CROSS.sources()}
        reruns = CROSS.policy_lag_rerun_sources()
        self.assertEqual({source.treatment for source in reruns}, {"none", "zero-loss"})
        self.assertTrue(historical.isdisjoint(source.namespace for source in reruns))
        self.assertTrue(all(source.history_path is None for source in reruns))

    def test_high_staleness_sources_preserve_queue_provenance(self):
        sources = [source for source in CROSS.sources() if "s32-40" in source.namespace]
        self.assertEqual(len(sources), 2)
        self.assertEqual({source.treatment for source in sources}, {"zero-loss", "staleness-aware"})
        self.assertTrue(all(source.cohort == "high-staleness-tbq8000" for source in sources))
        self.assertNotEqual(CROSS.color(32), CROSS.color(40))
        self.assertTrue(all(CROSS.color(age).startswith("#") for age in (32, 40)))

    def test_seven_staleness_legend_entries_wrap_inside_single_column(self):
        source = CROSS.Source("fixture", "staleness-aware", 16384, "fixture", None, None)
        histories = {
            CROSS.identity(source, f"s{age}-t1r7"): {
                "rollout/raw_reward": {1: 0.5, 2: 0.6},
                "staleness/total/mean": {1: 0.0, 2: float(age)},
            }
            for age in (12, 16, 20, 24, 28, 32, 40)
        }
        root = ET.fromstring(CROSS.render_training_group("staleness-aware", 16384, histories, {}))
        legends = [node for node in root.iter() if node.attrib.get("class") == "legend"]
        self.assertEqual(len(legends), 7)
        self.assertEqual(len({node.attrib["y"] for node in legends}), 2)
        self.assertTrue(all(float(node.attrib["x"]) + 45 < float(root.attrib["width"]) for node in legends))
        self.assertIn("Realized staleness", "".join(root.itertext()))
        self.assertIn("not evaluated", "".join(root.itertext()))

    def test_early_aime_endpoint_is_not_extended_through_unevaluated_steps(self):
        source = CROSS.Source("fixture", "zero-loss", 16384, "fixture", None, None)
        identity = CROSS.identity(source, "s32-t1r7")
        histories = {identity: {"rollout/raw_reward": {1: 0.5, 148: 0.75}}}
        downstream = {identity: {CROSS.AIME_MEAN_METRIC: {40: 48, 50: 49}}}
        root = ET.fromstring(CROSS.render_training_group("zero-loss", 16384, histories, downstream))
        self.assertIn("@50", "".join(root.itertext()))
        self.assertFalse(any(node.attrib.get("class") == "endpoint-leader" for node in root.iter()))


if __name__ == "__main__":
    unittest.main()
