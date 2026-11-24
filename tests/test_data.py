import tempfile
import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from ids.data import deduplicate, load_sample
from ids.preprocess import Preprocessor
from ids.schema import FEATURE_SPECS, label_value, resolve_columns


class DataTests(unittest.TestCase):
    def test_cic_header_versions_match_and_identifiers_excluded(self):
        a = resolve_columns([" " + spec[1] for spec in FEATURE_SPECS] + ["Source IP", "Destination Port", "Label"])
        b = resolve_columns([spec[2] for spec in FEATURE_SPECS])
        self.assertEqual(set(b), {spec[0] for spec in FEATURE_SPECS})
        self.assertTrue(set(b).issubset(a))
        self.assertNotIn("src_ip", b)
        self.assertNotIn("dst_port", b)

    def test_documented_labels_and_unknown_rejection(self):
        for label in ["BENIGN", " Benign "]:
            self.assertEqual(label_value(label), 0)
        for label in ["DDoS", "Web Attack – XSS", "Infilteration", "DDOS attack-HOIC", "DDOS attack-LOIC-UDP", "FTP-BruteForce"]:
            self.assertEqual(label_value(label), 1, label)
        self.assertIsNone(label_value("not a real label"))
        self.assertIsNone(label_value(""))

    def test_reservoir_spans_files_cleaning_and_embedded_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            specs = FEATURE_SPECS[:15]
            headers = [x[1] for x in specs]
            for file_index in range(2):
                frame = pd.DataFrame(np.arange(300 * 15).reshape(300, 15) + file_index * 10000, columns=headers)
                frame["Label"] = "BENIGN" if file_index == 0 else "DDoS"
                frame = frame.astype(str)
                frame.loc[0, headers[0]] = "Infinity"
                frame.loc[1, headers[0]] = "-1"
                frame.loc[len(frame)] = frame.columns
                frame.to_csv(path / f"part{file_index}.csv", index=False)
            sampled, features, report = load_sample(path, max_rows=100, chunk_size=37, log=lambda s: None)
            self.assertEqual(len(sampled), 100)
            self.assertEqual(sampled["target"].nunique(), 2)
            self.assertEqual(report["source_rows_scanned"], 602)
            self.assertEqual(report["embedded_headers_removed"], 2)
            self.assertTrue(np.isfinite(sampled[features].to_numpy()[~np.isnan(sampled[features].to_numpy())]).all())
            again, _, _ = load_sample(path, max_rows=100, chunk_size=37, log=lambda s: None)
            pd.testing.assert_frame_equal(sampled, again)
            # A later chunk keeps the original CSV data-row number.
            row = sampled.iloc[-1]
            original = pd.read_csv(path / row["source_file"])
            source_value = pd.to_numeric(pd.Series([original.iloc[int(row["source_row"]) - 2, 1]]))[0]
            self.assertEqual(row[features[1]], source_value)

    def test_unknown_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame({spec[1]: [1, 2] for spec in FEATURE_SPECS[:12]})
            frame["Label"] = ["BENIGN", "unexpected"]
            frame.to_csv(Path(temporary) / "bad.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                load_sample(temporary, max_rows=100, log=lambda s: None)

    def test_relative_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame({spec[1]: [1, 2, 3] for spec in FEATURE_SPECS[:12]})
            frame["Label"] = ["BENIGN", "DDoS", "BENIGN"]
            frame.to_csv(Path(temporary) / "flows.csv", index=False)
            relative = os.path.relpath(temporary)
            sampled, _, _ = load_sample(relative, max_rows=100, log=lambda s: None)
            self.assertEqual(len(sampled), 3)
            self.assertEqual(set(sampled["source_file"]), {"flows.csv"})

    def test_bidirectional_connection_metadata_groups_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame({spec[1]: [1, 2, 3] for spec in FEATURE_SPECS[:12]})
            frame["Label"] = ["BENIGN", "BENIGN", "DDoS"]
            frame["Source IP"] = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
            frame["Destination IP"] = ["10.0.0.2", "10.0.0.1", "10.0.0.4"]
            frame["Source Port"] = [1234, 443, 9999]
            frame["Destination Port"] = [443, 1234, 80]
            frame["Protocol"] = 6
            frame.to_csv(Path(temporary) / "groups.csv", index=False)
            sampled, _, report = load_sample(temporary, max_rows=100, log=lambda s: None)
            groups = sampled.sort_values("source_row")["_group"].to_numpy()
            self.assertTrue(report["grouping_available"])
            self.assertEqual(groups[0], groups[1])
            self.assertNotEqual(groups[0], groups[2])

    def test_duplicate_and_contradictory_feature_vectors(self):
        frame = pd.DataFrame({"a": [1., 1., 2., 2., 3.], "b": [4., 4., 5., 5., 6.],
                              "target": [0, 0, 0, 1, 1]})
        clean, report = deduplicate(frame, ["a", "b"])
        self.assertEqual(len(clean), 2)
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["contradictory_rows_removed"], 2)

    def test_preprocessing_training_only_and_roundtrip(self):
        train = np.array([[1., 2., np.nan], [3., np.nan, np.nan], [5., 4., np.nan]])
        processor = Preprocessor().fit(train)
        before = processor.median.copy()
        transformed = processor.transform(np.array([[1e20, np.nan, 1.]]))
        np.testing.assert_array_equal(processor.median, before)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertLessEqual(abs(transformed).max(), 20)
        self.assertEqual(processor.median[1], 3.)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preprocess.npz"
            processor.save(path)
            np.testing.assert_array_equal(processor.transform(train), Preprocessor.load(path).transform(train))


if __name__ == "__main__":
    unittest.main()
