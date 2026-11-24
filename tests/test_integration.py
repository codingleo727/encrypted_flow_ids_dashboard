"""Real model training on synthetic fixtures; no benchmark assertions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
import numpy as np
import pandas as pd

HAS_MODELS = all(importlib.util.find_spec(m) for m in ("torch", "xgboost", "streamlit"))


@unittest.skipUnless(HAS_MODELS, "Install requirements.txt to run real model integration checks")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.make_synthetic_fixture import make_fixture
        from ids.pipeline import DEFAULT_CONFIG, train, evaluate_external
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        make_fixture(cls.root / "input", rows=500)
        cls.run = cls.root / "run"
        config = {**DEFAULT_CONFIG, "max_rows": 1000, "chunk_size": 73, "epochs": 2,
                  "trees": 12, "threads": 1, "batch_size": 64, "demo_rows": 100,
                  "xgb_trials": 2, "cnn_trials": 2}
        train(cls.root / "input" / "2017", cls.run, config, log=lambda s: None)
        # Add an exact 2017 vector to external data; it must be removed.
        f17 = next((cls.root / "input" / "2017").glob("*.csv"))
        f18 = next((cls.root / "input" / "2018").glob("*.csv"))
        a, b = pd.read_csv(f17), pd.read_csv(f18)
        a.columns = b.columns
        pd.concat([b, a.iloc[:10]], ignore_index=True).to_csv(f18, index=False)
        cls.model_files = [cls.run / p for p in ("cnn.pt", "xgboost.json", "preprocess.npz", "decisions.json")]
        cls.before_external = [hashlib.sha256(f.read_bytes()).hexdigest() for f in cls.model_files]
        evaluate_external(cls.root / "input" / "2018", cls.run, max_rows=1000, chunk_size=73, log=lambda s: None)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_saved_models_reproduce_predictions(self):
        from ids.inference import PredictionBundle
        from ids.pipeline import manifest_for
        from ids.data import load_sample, deduplicate
        manifest = manifest_for(self.run)
        frame, _, _ = load_sample(self.root / "input" / "2017", features=manifest["features"], max_rows=1000,
                                   chunk_size=73, log=lambda s: None)
        frame, _ = deduplicate(frame, manifest["features"])
        actual = pd.read_csv(self.run / "2017_test" / "predictions.csv.gz")
        ordered = actual[["source_file", "source_row"]].merge(frame, on=["source_file", "source_row"], how="left", sort=False)
        x = ordered[manifest["features"]].to_numpy(dtype="float32")
        bundle = PredictionBundle.load(self.run)
        for name in bundle.names:
            scores = bundle.predict(x, name)
            np.testing.assert_allclose(scores, actual[f"{name}_score"], atol=1e-6)
            np.testing.assert_array_equal(scores >= bundle.threshold(name), actual[f"{name}_prediction"])

    def test_split_disjoint_and_external_overlap_removed(self):
        split = pd.read_csv(self.run / "splits.csv.gz", dtype={"_fingerprint": "uint64"})
        self.assertEqual(split["_fingerprint"].nunique(), len(split))
        self.assertEqual(set(split["partition"]), {"train", "validation", "test"})
        report = json.loads((self.run / "2018_unseen" / "data_report.json").read_text())
        self.assertEqual(report["cross_year_exact_overlaps_removed"], 10)
        self.assertEqual(report["evaluated_rows"], 500)

    def test_external_evaluation_never_changes_model_artifacts(self):
        from ids.pipeline import evaluate_external
        after = [hashlib.sha256(f.read_bytes()).hexdigest() for f in self.model_files]
        self.assertEqual(self.before_external, after)
        with self.assertRaisesRegex(ValueError, "already exists"):
            evaluate_external(self.root / "input" / "2018", self.run, max_rows=1000, log=lambda s: None)
        self.assertEqual(self.before_external, [hashlib.sha256(f.read_bytes()).hexdigest() for f in self.model_files])

    def test_streaming_prediction_without_label_column(self):
        from ids.predict import predict_csv
        source = next((self.root / "input" / "2018").glob("*.csv"))
        frame = pd.read_csv(source).iloc[:20].drop(columns="Label")
        frame.loc[len(frame)] = np.nan
        unlabeled = self.root / "unlabeled.csv"
        frame.to_csv(unlabeled, index=False)
        output = self.root / "predictions.csv"
        rows = predict_csv(unlabeled, self.run, output, chunk_size=7, log=lambda s: None)
        predicted = pd.read_csv(output)
        self.assertEqual(rows, 21)
        self.assertEqual(predicted.iloc[-1]["xgboost_prediction"], "Invalid")
        self.assertTrue(predicted.iloc[:20]["cnn_score"].between(0, 1).all())

    def test_streamlit_demo_controls(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"))
        app.run(timeout=30)
        app.text_input[0].set_value(str(self.run)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "100")
        app.selectbox[0].select("2018 unseen evaluation").run(timeout=30)
        app.selectbox[1].select("1D-CNN").run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "100")
        app.selectbox[1].select("Ensemble").run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "100")

    def test_full_sample_retraining_and_validation_roles(self):
        training = json.loads((self.run / 'training.json').read_text())
        split = pd.read_csv(self.run / 'splits.csv.gz')
        self.assertEqual(training['final_train_rows'], int((split['partition'] == 'train').sum()))
        self.assertLess(training['tuning_rows'], training['final_train_rows'])
        self.assertEqual(set(split.loc[split['partition'] == 'validation', 'validation_role']),
                         {'search', 'calibration', 'decision'})
        self.assertFalse(split.loc[split['partition'] != 'train', 'tuning_train'].any())
        trials = json.loads((self.run / 'trials.json').read_text())
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(row['status'] == 'complete' for row in trials))
        external = json.loads((self.run / '2018_unseen' / 'metrics.json').read_text())
        self.assertEqual(set(external), {'xgboost', 'cnn', 'ensemble'})


if __name__ == "__main__":
    unittest.main()
