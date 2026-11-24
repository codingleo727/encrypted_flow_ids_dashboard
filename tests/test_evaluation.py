import unittest
import numpy as np
import pandas as pd
from ids.metrics import binary_metrics
from ids.pipeline import DEFAULT_CONFIG, split_data, validate_config


class EvaluationTests(unittest.TestCase):
    def test_metric_denominators_and_single_class_handling(self):
        metrics = binary_metrics([0, 0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9, 0.1])
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["fpr"], 1/3)
        self.assertEqual([metrics[k] for k in ("tn", "fp", "fn", "tp")], [2, 1, 1, 1])
        benign_only = binary_metrics([0, 0], [0.1, 0.9])
        self.assertIsNone(benign_only["recall"])
        self.assertIsNone(benign_only["roc_auc"])
        attacks_only = binary_metrics([1, 1], [0.1, 0.9])
        self.assertIsNone(attacks_only["fpr"])

    def test_connection_group_splits(self):
        groups = np.repeat(np.arange(100), 4)
        frame = pd.DataFrame({"target": np.tile([0, 1, 0, 1], 100), "_group": groups})
        partitions, report = split_data(frame, DEFAULT_CONFIG)
        sets = {k: set(frame.iloc[v]["_group"]) for k, v in partitions.items()}
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertEqual(report["method"], "undirected_five_tuple_groups")

    def test_stratified_split_reproducible_and_complete(self):
        frame = pd.DataFrame({"target": np.tile([0, 0, 0, 1], 100)})
        partitions, report = split_data(frame, DEFAULT_CONFIG)
        again, _ = split_data(frame, DEFAULT_CONFIG)
        total = np.concatenate(list(partitions.values()))
        self.assertEqual(len(set(total)), len(frame))
        for key, indices in partitions.items():
            np.testing.assert_array_equal(indices, again[key])
            self.assertEqual(frame.iloc[indices]["target"].nunique(), 2)
        self.assertEqual(report["counts"]["train"]["n"], 240)
        self.assertEqual(report["counts"]["test"]["n"], 80)

    def test_invalid_resource_configuration_rejected(self):
        for change in [{"max_rows": 0}, {"threads": -1}, {"epochs": 0}, {"test_fraction": 0.9}]:
            with self.assertRaises(ValueError):
                validate_config({**DEFAULT_CONFIG, **change})


if __name__ == "__main__":
    unittest.main()
