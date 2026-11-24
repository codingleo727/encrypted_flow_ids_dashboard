import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from ids.preprocess import ModelPreprocessor
from ids.selection import (pr_auc, optimal_f1_threshold, tuning_subset, rare_label,
                           validation_parts, choose_decisions, blend_scores,
                           fit_sigmoid, apply_sigmoid, search_candidates, positive_weight)
from ids.pipeline import validate_config, DEFAULT_CONFIG


class SelectionTests(unittest.TestCase):
    def test_search_reproducible_unique_and_bounded(self):
        for model, count in [('xgboost', 18), ('cnn', 14)]:
            a = search_candidates(model, count)
            self.assertEqual(a, search_candidates(model, count))
            self.assertEqual(len({str(p) for p in a}), count)
            self.assertEqual(len(search_candidates(model, 81)), 81)
            with self.assertRaises(ValueError):
                search_candidates(model, 82)

    def test_training_subsample_preserves_rare_families(self):
        labels = ['BENIGN'] * 100 + ['DDoS'] * 80 + ['Heartbleed'] * 3 + ['FTP-Patator'] * 7 + ['Web Attack – XSS'] * 2
        frame = pd.DataFrame({'attack_label': labels})
        selected = tuning_subset(frame)
        np.testing.assert_array_equal(selected, tuning_subset(frame))
        self.assertEqual(len(selected), 25 + 20 + 3 + 7 + 2)
        self.assertTrue(set(range(180, 192)).issubset(selected))
        self.assertTrue(rare_label('Infilteration'))
        self.assertFalse(rare_label('PortScan'))

    def test_validation_roles_are_disjoint_complete(self):
        frame = pd.DataFrame({'target': np.tile([0, 0, 0, 1], 100)})
        parts = validation_parts(frame)
        self.assertEqual([len(parts[k]) for k in ['search', 'calibration', 'decision']], [200, 100, 100])
        self.assertEqual(len(set(np.concatenate(list(parts.values())))), len(frame))
        for key in parts:
            np.testing.assert_array_equal(parts[key], validation_parts(frame)[key])

    def test_validation_roles_preserve_connections(self):
        frame = pd.DataFrame({'target': np.tile([0, 1], 200), '_group': np.repeat(np.arange(100), 4)})
        parts = validation_parts(frame)
        groups = {k: set(frame.iloc[v]['_group']) for k, v in parts.items()}
        self.assertFalse(groups['search'] & groups['decision'])
        self.assertFalse(groups['calibration'] & groups['decision'])
        self.assertFalse(groups['search'] & groups['calibration'])

    def test_too_small_validation_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'at least 8'):
            validation_parts(pd.DataFrame({'target': [0]*20 + [1]*7}))

    def test_preprocessing_mask_uses_training_only(self):
        x = np.array([[1, 7, np.nan, 4], [3, 7, np.nan, np.nan], [5, 7, np.nan, 8]], dtype=float)
        p = ModelPreprocessor().fit(x)
        np.testing.assert_array_equal(p.active, [True, False, False, True])
        raw = p.transform(x, 'xgboost')
        np.testing.assert_allclose(raw[:, 0], x[:, 0])
        self.assertTrue(np.isnan(raw[1, 1]))
        np.testing.assert_allclose(p.transform(x, 'cnn')[:, 0], [-1, 0, 1])
        before = p.median.copy()
        p.transform(np.array([[100000, 100, 10, 200000]]), 'cnn')
        np.testing.assert_array_equal(p.median, before)
        self.assertEqual(p.transform(x).shape[1], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pre.npz'
            p.save(path)
            restored = ModelPreprocessor.load(path)
            np.testing.assert_array_equal(p.transform(x), restored.transform(x))

    def test_f1_threshold_matches_brute_force(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0.03, 0.12, 0.1, 0.19, 0.2, 0.01])
        threshold = optimal_f1_threshold(y, scores)
        self.assertNotEqual(threshold, 0.5)
        best = max(f1_score(y, scores >= t) for t in np.unique(scores))
        self.assertAlmostEqual(f1_score(y, scores >= threshold), best)

    def test_pr_auc_is_explicitly_not_average_precision(self):
        y = [0, 1, 0, 1]
        scores = [0.7, 0.6, 0.4, 0.2]
        self.assertNotAlmostEqual(pr_auc(y, scores), average_precision_score(y, scores))
        self.assertEqual(pr_auc([0, 1], [0.1, 0.9]), 1.0)
        with self.assertRaises(ValueError):
            pr_auc([0, 0], [0.1, 0.5])

    def test_weight_formulas_use_actual_training_prevalence(self):
        y = [0] * 9 + [1]
        self.assertEqual(positive_weight(y, 'one'), 1)
        self.assertEqual(positive_weight(y, 'sqrt_ratio'), 3)
        self.assertEqual(positive_weight(y, 'ratio'), 9)

    def test_ensemble_grid_contains_endpoints_and_keeps_best_base(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        a = np.array([.1, .9, .2, .8, .3, .7, .4, .6])
        b = 1 - a
        scores = {'xgboost': a, 'cnn': b}
        settings = choose_decisions(y, scores, y, scores)
        self.assertEqual(len(settings['raw_weight_search']), 11)
        prediction = blend_scores(a, b, settings['ensemble'])
        self.assertGreaterEqual(pr_auc(y, prediction), max(pr_auc(y, a), pr_auc(y, b)))
        self.assertEqual(set(settings['thresholds']), {'xgboost', 'cnn', 'ensemble'})

    def test_sigmoid_serializable_and_finite_at_extremes(self):
        y = [0, 0, 1, 1]
        params = fit_sigmoid(y, [.1, .3, .7, .9])
        import json
        copied = json.loads(json.dumps(params))
        a = apply_sigmoid(np.array([0, .2, .8, 1]), copied)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((np.diff(a) > 0).all())

    def test_invalid_tuning_budget(self):
        for changes in [{'xgb_trials': 82}, {'cnn_trials': 0}, {'tuning_majority_fraction': 0}]:
            with self.assertRaises(ValueError):
                validate_config({**DEFAULT_CONFIG, **changes})


if __name__ == '__main__':
    unittest.main()
