"""CPU-only coverage checks for the historical/high-staleness TIS figures."""

import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch


PLOT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "experiments/outputs/reasoning_eval/wandb-collapse-20260901"
)
sys.path.insert(0, str(PLOT_ROOT))
SPEC = importlib.util.spec_from_file_location("tis_diagnostic_figures_test", PLOT_ROOT / "plot_tis_diagnostics.py")
TIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TIS
SPEC.loader.exec_module(TIS)
sys.path.pop(0)


class TisDiagnosticFigureTests(unittest.TestCase):
    def test_high_staleness_sources_include_both_treatments_and_queue_size(self):
        high = [source for source in TIS.SOURCES if "s32-40" in source.namespace]
        self.assertEqual({source.treatment for source in high}, {"zero-loss", "staleness-aware"})
        self.assertTrue(all(source.training_buffer_queue_size == 8000 for source in high))
        self.assertTrue(any("zero-loss-trunc-s24-28" in source.namespace for source in TIS.SOURCES))
        self.assertFalse(any("policy-lag-v1" in source.namespace for source in TIS.SOURCES))

    def test_discovery_keeps_high_staleness_and_rejects_independent_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            high = next(source for source in TIS.SOURCES if source.treatment == "zero-loss" and "s32-40" in source.namespace)
            for age in (32, 40):
                dashboard = root / f"max-weight-staleness-{age}-from-prefill" / f"s{age}-t1r7-{high.namespace}-suffix" / "dump/dashboard/metrics.jsonl"
                dashboard.parent.mkdir(parents=True)
                dashboard.touch()
            with patch.object(TIS, "TRAINING_ROOT", root), patch.object(TIS, "SOURCES", (high,)):
                arms = TIS.discover_arms()
                self.assertEqual([arm.staleness for arm in arms], [32, 40])
                self.assertTrue(all(arm.training_buffer_queue_size == 8000 for arm in arms))
                duplicate = TIS.Source("independent-rerun", "zero-loss", "zero loss on truncated")
                path = root / "max-weight-staleness-32-from-prefill/s32-t1r7-independent-rerun-suffix/dump/dashboard/metrics.jsonl"
                path.parent.mkdir(parents=True)
                path.touch()
                with patch.object(TIS, "SOURCES", (high, duplicate)):
                    with self.assertRaisesRegex(ValueError, "independent runs share plot identity"):
                        TIS.discover_arms()

    def test_high_staleness_curves_are_drawn_without_extending_their_endpoints(self):
        histories = {
            (treatment, age): {
                "train/tis_abs": {1: 0.01, 80: 0.03},
                "staleness/total/mean": {1: 0.0, 80: float(age)},
            }
            for treatment in ("zero-loss", "staleness-aware")
            for age in (32, 40)
        }
        root = ET.fromstring(TIS.render_cross_treatment(histories))
        groups = [node for node in root.iter() if node.attrib.get("data-metric") == "train/tis_abs"]
        self.assertEqual(len(groups), 4)
        for group in groups:
            curves = [node for node in group if node.attrib.get("class") == "smooth"]
            self.assertEqual(len(curves), 1)
            points = curves[0].attrib["points"].split()
            self.assertEqual(len(points), 2)
            first_x = float(points[0].split(",")[0])
            last_x = float(points[-1].split(",")[0])
            self.assertAlmostEqual(last_x - first_x, (390 - 82) * (80 - 1) / (300 - 1), places=1)
        text = "".join(root.itertext())
        self.assertIn("S=32", text)
        self.assertIn("S=40", text)
        self.assertIn("queue 8000", text)
        self.assertIn("queue 6000", text)

    def test_staleness_axis_does_not_flatten_32_and_40_at_28(self):
        spec = next(spec for spec in TIS.metric_specs(40) if spec.name == "staleness/total/mean")
        self.assertEqual(spec.upper, 40)
        self.assertEqual(spec.ticks, (0, 10, 20, 30, 40))
        self.assertLess(TIS.scale_value(spec, 28), TIS.scale_value(spec, 32))
        self.assertLess(TIS.scale_value(spec, 32), TIS.scale_value(spec, 40))

    def test_legend_is_separate_from_header_notes_and_column_titles(self):
        histories = {("zero-loss", age): {} for age in (8, 12, 16, 20, 24, 28, 32, 40)}
        root = ET.fromstring(TIS.render_cross_treatment(histories))
        labels = [node for node in root.iter() if node.attrib.get("class") == "legend"]
        notes = [node for node in root.iter() if node.attrib.get("class") == "note"]
        self.assertTrue(all(float(node.attrib["y"]) > max(float(note.attrib["y"]) for note in notes) + 14 for node in labels))
        self.assertTrue(all(float(node.attrib["x"]) < float(root.attrib["width"]) - 70 for node in labels))

    def test_summary_keeps_queue_provenance_and_actual_tis_endpoint(self):
        arm = TIS.Arm("fixture", "zero-loss", "zero loss", "s32-t1r7", 32, Path("unused"), 8000)
        history = {("zero-loss", 32): {"train/tis_abs": {10: 0.02}, "rollout/raw_reward": {11: 0.8}}}
        row = TIS.summary_rows([arm], history)[0]
        self.assertEqual(row["training_buffer_queue_size"], 8000)
        self.assertEqual(row["last_observed_step"], 11)
        self.assertEqual(row["last_tis_step"], 10)


if __name__ == "__main__":
    unittest.main()
