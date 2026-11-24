"""Separate training and locked-model external evaluation workflows."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import deduplicate, fingerprints, load_sample
from .metrics import benchmark, binary_metrics, per_attack_metrics
from .preprocess import ModelPreprocessor
from .selection import choose_decisions, tuning_subset, validation_parts


DEFAULT_CONFIG = {"max_rows": 150000, "chunk_size": 25000, "seed": 42,
                  "threads": 4, "trees": 1500, "epochs": 35, "patience": 6,
                  "xgb_patience": 30, "xgb_trials": 18, "cnn_trials": 14,
                  "tuning_majority_fraction": 0.25, "cnn_epoch_budget_seconds": 180,
                  "batch_size": 512, "test_fraction": 0.2, "validation_fraction": 0.2,
                  "demo_rows": 2000}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def manifest_for(run):
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "trained":
        raise ValueError("This run did not finish training. Use a completed run.")
    return manifest


def validate_config(config):
    for k in ("max_rows", "chunk_size", "threads", "trees", "epochs", "patience", "batch_size", "demo_rows", "xgb_patience", "xgb_trials", "cnn_trials", "cnn_epoch_budget_seconds"):
        if not isinstance(config[k], int) or isinstance(config[k], bool) or config[k] < 1:
            raise ValueError(f"{k} must be a positive integer.")
    if not 0 < config["tuning_majority_fraction"] <= 1:
        raise ValueError("tuning_majority_fraction must be in (0, 1].")
    if config["xgb_trials"] > 81 or config["cnn_trials"] > 81:
        raise ValueError("Each search space contains 81 unique primary configurations.")
    if config["max_rows"] < 100:
        raise ValueError("max_rows must be at least 100.")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not 0 < config["test_fraction"] < 0.5 or not 0 < config["validation_fraction"] < 0.5:
        raise ValueError("Test and validation fractions must each be between 0 and 0.5.")
    if config["test_fraction"] + config["validation_fraction"] >= 0.8:
        raise ValueError("Reserve at least 20% for training.")


def split_data(frame, config):
    y = frame["target"].to_numpy()
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 10:
        raise ValueError("Need at least 10 unique benign and 10 unique attack flows. "
                         "Include attack-day CSVs and/or increase max_rows.")
    idx = np.arange(len(frame))
    grouped = "_group" in frame
    def split(indices, fraction, seed):
        if not grouped:
            return train_test_split(indices, test_size=fraction, random_state=seed, stratify=y[indices])
        groups = frame["_group"].to_numpy()[indices]
        # Group by undirected five-tuple when metadata exists. Retry only for
        # class support, never according to held-out performance.
        splitter = GroupShuffleSplit(n_splits=40, test_size=fraction, random_state=seed)
        for a, b in splitter.split(indices, y[indices], groups):
            if all(len(np.unique(y[indices[part]])) == 2 and
                   min(np.bincount(y[indices[part]])) >= 2 for part in (a, b)):
                return indices[a], indices[b]
        raise ValueError("Could not create disjoint connection groups with both classes. "
                         "Include more independent connections/CSV files.")
    remaining, test = split(idx, config["test_fraction"], config["seed"])
    train, val = split(remaining, config["validation_fraction"] / (1 - config["test_fraction"]), config["seed"] + 1)
    return {"train": train, "validation": val, "test": test}, {
        "method": "undirected_five_tuple_groups" if grouped else "stratified_random_after_exact_vector_deduplication",
        "limitation": "Related flows from the same host/session/capture can still cross partitions; "
                      "within-2017 results do not establish temporal or deployment generalization.",
        "requested_test_fraction": config["test_fraction"],
        "requested_validation_fraction": config["validation_fraction"],
        "counts": {k: {"n": len(v), "benign": int((y[v] == 0).sum()), "attacks": int((y[v] == 1).sum())}
                   for k, v in {"train": train, "validation": val, "test": test}.items()},
    }


def save_demo(frame, features, path, n, seed):
    # Independent, uniformly selected preview; metrics for it are separate from
    # the full evaluated sample. Sampling never uses predicted labels.
    frame = frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)
    np.savez_compressed(path, x=frame[features].to_numpy(dtype="float32"),
                        y=frame["target"].to_numpy(dtype="int8"),
                        source_file=frame["source_file"].to_numpy(dtype=str),
                        source_row=frame["source_row"].to_numpy(dtype="int64"),
                        attack_label=frame["attack_label"].to_numpy(dtype=str))


def evaluate_models(bundle, frame, features, directory, dataset, log=print):
    output = directory / dataset
    output.mkdir()
    raw = frame[features].to_numpy(dtype="float32")
    y = frame["target"].to_numpy(dtype="int8")
    metrics = {}
    predictions = frame[["source_file", "source_row", "attack_label", "target"]].reset_index(drop=True).copy()
    attacks = []
    for name in bundle.names:
        log(f"Evaluating {name} on {dataset}: {len(frame):,} flows...")
        predict = lambda a, name=name: bundle.predict(a, name)
        scores = predict(raw)
        threshold = bundle.threshold(name)
        metrics[name] = binary_metrics(y, scores, threshold)
        timing = benchmark(predict, raw)
        timing["inference_us_per_flow"] = timing.pop("model_us_per_flow")
        timing["latency_scope"] = "warmed CPU inference including preprocessing, calibration and blending as applicable; excludes loading and I/O"
        metrics[name].update(timing)
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_prediction"] = (scores >= threshold).astype("int8")
        for row in per_attack_metrics(frame, scores, threshold):
            attacks.append({"model": name, **row})
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(attacks).to_csv(output / "attack_breakdown.csv", index=False)
    write_json(output / "metrics.json", metrics)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(output / "metrics.csv", index_label="model")
    return metrics


def train(data2017, run, config, log=print):
    from .models import fit_xgb, fit_cnn, save_models
    from .tuning import search_models
    from .inference import PredictionBundle
    validate_config(config)
    config = config.copy()
    import os
    config["threads"] = min(config["threads"], os.cpu_count() or 1)
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise ValueError(f"{run} is not empty. Choose a new --run folder to preserve existing experiments.")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "manifest.json", {"status": "training", "config": config})
    started = perf_counter()
    frame, features, ingest = load_sample(data2017, max_rows=config["max_rows"],
        chunk_size=config["chunk_size"], seed=config["seed"], log=log)
    frame, duplicates = deduplicate(frame, features)
    partitions, splitting = split_data(frame, config)
    a, b = partitions["train"], partitions["validation"]
    validation = frame.iloc[b].reset_index(drop=True)
    roles = validation_parts(validation, config["seed"])
    splitting["validation_subsets"] = {k: {"n": len(v), "benign": int((validation.iloc[v]["target"] == 0).sum()),
                                         "attacks": int((validation.iloc[v]["target"] == 1).sum())} for k, v in roles.items()}
    training_frame = frame.iloc[a].reset_index(drop=True)
    subset = tuning_subset(training_frame, config["tuning_majority_fraction"], config["seed"])
    write_json(run / "data2017.json", {**ingest, **duplicates, "retained_unique_rows": len(frame),
        "sampled_label_counts": frame["attack_label"].value_counts().to_dict(),
        "rare_preservation_scope": "all rare examples already in sampled training partition; original uniform sampling may omit rare labels"})
    supports = []
    for label, count in sorted(ingest["eligible_label_counts"].items()):
        from .selection import rare_label
        row = {"label": label, "rare": rare_label(label), "eligible_source_rows": count,
               "retained_sample": int(frame["attack_label"].eq(label).sum()),
               "train": int(training_frame["attack_label"].eq(label).sum()),
               "tuning_train": int(training_frame.iloc[subset]["attack_label"].eq(label).sum()),
               "test": int(frame.iloc[partitions["test"]]["attack_label"].eq(label).sum())}
        for role, indices in roles.items():
            row["validation_" + role] = int(validation.iloc[indices]["attack_label"].eq(label).sum())
        supports.append(row)
        if row["rare"] and (row["train"] == 0 or row["validation_search"] < 5):
            log(f"Rare-label support warning: {label}: train={row['train']}, "
                f"search-validation={row['validation_search']}. Recall may be unmeasurable or unstable.")
    pd.DataFrame(supports).to_csv(run / "label_support.csv", index=False)
    write_json(run / "split.json", splitting)
    assignment = frame[["source_file", "source_row", "attack_label", "target", "_fingerprint"]].copy()
    assignment["partition"], assignment["validation_role"] = "", ""
    for key, indices in partitions.items():
        assignment.loc[indices, "partition"] = key
    for key, indices in roles.items():
        assignment.loc[b[indices], "validation_role"] = key
    assignment["tuning_train"] = False
    assignment.loc[a[subset], "tuning_train"] = True
    assignment.to_csv(run / "splits.csv.gz", index=False, compression="gzip")
    log(f"Usable input features: {len(features)}. Split: {splitting['method']}.")
    log(str(splitting["counts"]))
    log(f"Tuning training sample: {len(subset):,}; full sampled training partition: {len(a):,}.")
    raw = frame[features].to_numpy(dtype="float32")
    y = frame["target"].to_numpy(dtype="int8")
    processor = ModelPreprocessor().fit(raw[a])
    processor.save(run / "preprocess.npz")
    search_indices = b[roles["search"]]
    xs = {name: processor.transform(raw[a[subset]], name) for name in ("xgboost", "cnn")}
    xv = {name: processor.transform(raw[search_indices], name) for name in xs}
    best_params, trials = search_models(xs, y[a[subset]], xv,
        frame.iloc[search_indices].reset_index(drop=True), config, run, log)
    write_json(run / "best_parameters.json", best_params)
    # Refit each winner once on ALL rows of the sampled training partition.
    log("Retraining both winners on the full sampled training partition...")
    models, fits = {}, {}
    for name, fit in (("xgboost", fit_xgb), ("cnn", fit_cnn)):
        extra = {"enforce_budget": False} if name == "cnn" else {}
        models[name], fits[name] = fit(processor.transform(raw[a], name), y[a], xv[name],
                                       y[search_indices], best_params[name], config, log, **extra)
    save_models(models, run)
    bundle = PredictionBundle(processor, models)
    scores = {}
    for role in ("calibration", "decision"):
        indices = b[roles[role]]
        scores[role] = {name: bundle.predict(raw[indices], name) for name in models}
    decisions = choose_decisions(y[b[roles["calibration"]]], scores["calibration"],
                                  y[b[roles["decision"]]], scores["decision"])
    write_json(run / "decisions.json", decisions)
    bundle.decisions = decisions
    training = {"selected_parameters": best_params, "final_fit": fits,
        "cnn_history": fits["cnn"]["history"], "tuning_rows": len(subset), "final_train_rows": len(a),
        "tuning_label_counts": training_frame.iloc[subset]["attack_label"].value_counts().to_dict(),
        "search_method": "seeded random without replacement, bounded diagnostic proposals within original trial budget",
        "primary_metric": "trapezoidal PR-AUC; average precision reported separately",
        "validation_reuse": "search subset reused across trials/early stopping; calibration subset separate; blend and F1 thresholds selected on decision subset"}
    write_json(run / "training.json", training)
    pd.DataFrame(training["cnn_history"]).to_csv(run / "cnn_history.csv", index=False)
    active_features = [f for f, keep in zip(features, processor.active) if keep]
    pd.DataFrame({"feature": active_features, "importance": models["xgboost"].feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(run / "feature_importance.csv", index=False)
    np.save(run / "reference_hashes.npy", frame["_fingerprint"].to_numpy(dtype="uint64"), allow_pickle=False)
    evaluate_models(bundle, validation, features, run, "2017_validation", log)
    test = frame.iloc[partitions["test"]].reset_index(drop=True)
    evaluate_models(bundle, test, features, run, "2017_test", log)
    save_demo(test, features, run / "demo_2017.npz", config["demo_rows"], config["seed"])
    manifest = {"status": "trained", "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": features, "active_features": active_features,
        "dropped_constant_features": [f for f in features if f not in active_features],
        "thresholds": decisions["thresholds"], "config": config,
        "representation": "grouped aggregate flow features; CNN position is feature order, not packet time",
        "encryption_scope": "payload-free statistics; encrypted-only membership not established",
        "source_kind": "synthetic_fixture" if any("SYNTHETIC" in Path(p["path"]).name.upper() for p in ingest["inventory"]) else "user_supplied_csv",
        "total_seconds": perf_counter() - started, "python": sys.version, "platform": platform.platform(),
        "dependencies": {k: version(k) for k in ("numpy", "pandas", "scikit-learn", "torch", "streamlit")},
        "xgboost_version": __import__("xgboost").__version__}
    write_json(run / "manifest.json", manifest)
    log(f"Training complete. Results saved in {run.resolve()}")
    return manifest


def evaluate_external(data2018, run, max_rows=150000, chunk_size=25000, log=print):
    from .inference import PredictionBundle
    run = Path(run)
    manifest = manifest_for(run)
    if (run / "2018_unseen").exists():
        raise ValueError("2018 evaluation already exists for this run. Keep it as the final external result; "
                         "use its saved results for the demo.")
    config = manifest["config"]
    features = manifest["features"]
    frame, _, ingest = load_sample(data2018, features=features, max_rows=max_rows,
                                   chunk_size=chunk_size, seed=config["seed"] + 100, log=log,
                                   group_connections=False)
    frame, duplicates = deduplicate(frame, features)
    reference = np.load(run / "reference_hashes.npy", allow_pickle=False)
    overlaps = np.isin(frame["_fingerprint"].to_numpy(dtype="uint64"), reference)
    overlap_count = int(overlaps.sum())
    frame = frame.loc[~overlaps].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No external rows remain after removing exact overlaps with 2017.")
    bundle = PredictionBundle.load(run)
    # Commit an external result only after every output has been produced.
    # A failed scan/model invocation does not lock out a corrected retry.
    with tempfile.TemporaryDirectory(prefix="external_staging_", dir=run) as temporary:
        stage = Path(temporary)
        metrics = evaluate_models(bundle, frame, features, stage, "2018_unseen", log)
        write_json(stage / "2018_unseen" / "data_report.json", {**ingest, **duplicates,
            "cross_year_exact_overlaps_removed": overlap_count, "evaluated_rows": len(frame),
            "scope": "uniform sample of the supplied 2018 files, then deduplicated and overlap-filtered; no model tuning"})
        save_demo(frame, features, stage / "demo_2018.npz", config["demo_rows"], config["seed"])
        (stage / "2018_unseen").rename(run / "2018_unseen")
        (stage / "demo_2018.npz").replace(run / "demo_2018.npz")
    log(f"External evaluation complete. Results saved in {run / '2018_unseen'}")
    return metrics
