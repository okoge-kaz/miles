"""CPU-only checks for plotting measured policy lag, not imputed diagnostics."""

import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.tools.reasoning_eval import plot_policy_lag as plot


def valid_row():
    row = {"policy_lag/loss_sensitivity_supported": 1}
    for population in plot.POPULATIONS:
        base = f"policy_lag/{population}/"
        row.update({base + "response_token_count": 10, base + "nonfinite_delta_count": 0,
                    base + "delta_rms": 0.1, base + plot.RETAINED: 0,
                    base + "loss_sensitivity_retained_fraction": 0})
        for stage in ("pre", "post"):
            row.update({base + plot.WEIGHTED + stage: 0.1,
                        base + f"nonfinite_loss_sensitivity_count_{stage}": 0,
                        base + f"loss_sensitivity_weighted_metrics_valid_{stage}": 1})
    return row


class PolicyLagFigureTests(unittest.TestCase):
    def test_missing_and_invalid_steps_break_curves(self):
        self.assertEqual(plot.segments([(1, 1), (2, math.nan), (3, 3), (5, 5), (6, 6)]),
                         [[(1, 1)], [(3, 3)], [(5, 5), (6, 6)]])

    def test_zero_sensitivity_retention_is_valid_but_weighted_rms_is_not(self):
        row = valid_row()
        row["policy_lag/all_response/loss_sensitivity_weighted_metrics_valid_post"] = 0
        self.assertTrue(math.isnan(plot.measurement(row, plot.WEIGHTED + "post")))
        self.assertEqual(plot.measurement(row, plot.RETAINED), 0)
        self.assertEqual(plot.measurement(row, "loss_sensitivity_retained_fraction"), 0)

    def test_unsupported_loss_does_not_hide_delta(self):
        row = valid_row()
        row["policy_lag/loss_sensitivity_supported"] = 0
        self.assertEqual(plot.measurement(row, "delta_rms"), 0.1)
        self.assertTrue(math.isnan(plot.measurement(row, plot.WEIGHTED + "pre")))

    def test_empty_or_nonfinite_population_is_not_a_zero(self):
        for field, value in (("response_token_count", 0), ("nonfinite_delta_count", 1)):
            row = valid_row()
            row[f"policy_lag/all_response/{field}"] = value
            self.assertTrue(math.isnan(plot.measurement(row, "delta_rms")))
            self.assertTrue(math.isnan(plot.measurement(row, plot.RETAINED)))

    def test_step_join_replay_and_partial_tail(self):
        records = [
            {"step_key": "rollout/step", "step": 0, "metrics": {"rollout/raw_reward": 0.4}},
            {"step_key": "train/step", "step": 0, "metrics": valid_row()},
            {"step_key": "train/step", "step": 0,
             "metrics": {"policy_lag/all_response/delta_rms": 0.2}},
            {"step_key": "rollout/step", "step": 1, "metrics": {"rollout/raw_reward": 0.5}},
            {"step_key": "train/step", "step": 2, "metrics": valid_row()},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records)
                            + '\n{"policy_lag/incomplete":', encoding="utf-8")
            rows, incomplete = plot.read_rows(path)
        self.assertEqual(incomplete, 1)
        self.assertEqual(set(rows), {1, 3})
        self.assertEqual(rows[1]["rollout/raw_reward"], 0.4)
        self.assertEqual(plot.measurement(rows[1], "delta_rms"), 0.2)
        self.assertNotIn("rollout/raw_reward", rows[3])

    def test_malformed_complete_record_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"policy_lag/broken":\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                plot.read_rows(path)

    def test_renderers_produce_finite_svg_and_keep_late_start(self):
        run = plot.Run(plot.COHORTS[4], 28, Path("fixture"), {251: valid_row(), 252: valid_row()})
        for svg in (plot.overview([run]), plot.scatter([run]), plot.populations([run]),
                    plot.populations([run], retention=True), plot.staleness_and_policy_lag_by_step([run])):
            root = ET.fromstring(svg)
            for node in root.iter():
                for name in ("points", "cx", "cy"):
                    self.assertNotIn("nan", node.attrib.get(name, "").lower())
            self.assertIn("28", "".join(root.itertext()))
        self.assertEqual(plot.points(run, "delta_rms"), [(251, 0.1), (252, 0.1)])

    def test_step_figure_aligns_two_metrics_without_dual_axes(self):
        first = {**valid_row(), plot.COMPARATORS[0]: 2}
        second = {**valid_row(), plot.COMPARATORS[0]: 4}
        run = plot.Run(plot.COHORTS[0], 8, Path("fixture"), {10: first, 11: second})
        root = ET.fromstring(plot.staleness_and_policy_lag_by_step([run]))
        texts = [node.text for node in root.iter() if node.tag.endswith("}text")]
        self.assertEqual(texts.count("Training step"), 6)
        self.assertEqual(texts.count("Realized staleness"), 3)
        self.assertEqual(texts.count("Policy lag RMS (nats)"), 3)
        self.assertNotIn("Raw reward", texts)
        curves = [node for node in root.iter() if node.tag.endswith("}polyline")]
        self.assertEqual(len(curves), 2)
        self.assertEqual(curves[0].attrib["stroke"], curves[1].attrib["stroke"])
        x_coordinates = [[point.split(",")[0] for point in curve.attrib["points"].split()] for curve in curves]
        self.assertEqual(x_coordinates[0], x_coordinates[1])


if __name__ == "__main__":
    unittest.main()
