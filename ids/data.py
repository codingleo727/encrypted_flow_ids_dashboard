"""Chunked CSV ingestion and uniform reservoir sampling across all files."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from .schema import FEATURES, label_value, normalize, resolve_columns


def csv_files(path):
    path = Path(path)
    files = [path] if path.is_file() and path.suffix.lower() == ".csv" else sorted(
        (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"),
        key=lambda p: p.as_posix().lower(),
    )
    if not files:
        raise ValueError(f"No CSV files found in {path}. Extract the dataset ZIP files first.")
    return files


def inspect_files(path):
    inventory = []
    for file in csv_files(path):
        # latin1 accepts the original 2017 attack-label separators as well as
        # ASCII CSVs. A possible UTF8 BOM is removed from the header below.
        columns = pd.read_csv(file, nrows=0, encoding="utf-8-sig", encoding_errors="replace").columns
        mapping = resolve_columns(columns)
        if "label" not in mapping:
            raise ValueError(f"{file.name}: missing Label column. Use labeled CIC flow CSV files.")
        inventory.append({"path": str(file.resolve()), "bytes": file.stat().st_size,
                          "features": [f for f in FEATURES if f in mapping],
                          "mapping": mapping})
    return inventory


def common_features(inventory, minimum=12):
    features = [f for f in FEATURES if all(f in item["features"] for item in inventory)]
    if len(features) < minimum:
        raise ValueError(f"Only {len(features)} compatible features across CSVs; need {minimum}. "
                         "Run 'python -m ids inspect --data PATH' to inspect headers.")
    return features


def clean_numeric(frame, features):
    numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    # Selected schema consists of nonnegative durations, sizes, rates, counts.
    numeric = numeric.mask((numeric < 0) | (numeric > np.finfo(np.float32).max))
    return numeric.astype("float32")


def fingerprints(frame, features):
    return pd.util.hash_pandas_object(frame[features], index=False).to_numpy(dtype="uint64")


def deduplicate(frame, features):
    """Drop contradictory labels for the same vector; retain one exact vector."""
    frame = frame.copy()
    frame["_fingerprint"] = fingerprints(frame, features)
    conflicts = frame.groupby("_fingerprint")["target"].transform("nunique") > 1
    contradictory = int(conflicts.sum())
    frame = frame.loc[~conflicts]
    before = len(frame)
    frame = frame.drop_duplicates("_fingerprint").reset_index(drop=True)
    return frame, {"contradictory_rows_removed": contradictory,
                   "duplicate_rows_removed": before - len(frame)}


def load_sample(path, features=None, max_rows=150000, chunk_size=25000, seed=42,
                log=print, inventory=None, group_connections=True):
    """Scan the whole supplied data; retain a bounded, label-independent sample.

    Assign each eligible row a uniform random priority and keep the smallest K.
    This avoids taking only early benign records or particular attack days.
    Raw string data is bounded by chunk_size; selected rows are float32.
    """
    if max_rows < 100 or chunk_size < 1:
        raise ValueError("max_rows must be at least 100 and chunk_size must be positive.")
    inventory = inventory or inspect_files(path)
    features = features or common_features(inventory)
    endpoint_fields = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    use_groups = group_connections and all(all(k in f["mapping"] for k in endpoint_fields) for f in inventory)
    rng = np.random.default_rng(seed)
    reservoir = None
    counts = Counter()
    total = repeated = invalid = 0
    unknown = Counter()
    for info in inventory:
        file = Path(info["path"])
        mapping = info["mapping"]
        missing = [f for f in features if f not in mapping]
        if missing:
            raise ValueError(f"{file.name}: missing model features: {', '.join(missing)}. "
                             "Use the original compatible CICFlowMeter CSV release.")
        wanted = features + ["label"] + (endpoint_fields if use_groups else [])
        selected = {mapping[k]: k for k in wanted}
        log(f"Reading {file.name} ({info['bytes'] / 1e6:.1f} MB)...")
        reader = pd.read_csv(file, usecols=list(selected), dtype=str, chunksize=chunk_size,
                             encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="error")
        for chunk_index, chunk in enumerate(reader):
            chunk = chunk.rename(columns=selected)
            total += len(chunk)
            labels = chunk["label"].fillna("").str.strip()
            is_header = labels.map(normalize).eq("label")
            repeated += int(is_header.sum())
            unique_map = {label: label_value(label) for label in labels.unique()}
            y = labels.map(unique_map)
            unknown.update(labels.loc[y.isna() & ~is_header].value_counts().to_dict())
            valid_label = y.notna() & ~is_header
            original = chunk.index.to_numpy()
            # Store a stable file-relative CSV data-row number for investigation.
            meta = pd.DataFrame({"source_file": file.relative_to(Path(path).resolve()).as_posix()
                                 if Path(path).is_dir() else file.name,
                                 "source_row": original + 2,
                                 "attack_label": labels.to_numpy(),
                                 "target": y.to_numpy()}, index=chunk.index)
            numeric = clean_numeric(chunk, features)
            has_values = numeric.notna().any(axis=1)
            invalid += int((valid_label & ~has_values).sum())
            keep = valid_label & has_values
            counts.update(labels.loc[keep].value_counts().to_dict())
            if not keep.any():
                continue
            batch = pd.concat([numeric.loc[keep], meta.loc[keep]], axis=1)
            batch["target"] = batch["target"].astype("int8")
            if use_groups:
                ep = chunk.loc[keep, endpoint_fields].fillna("").astype(str)
                if ep.apply(lambda col: col.str.strip().eq("")).any(axis=None):
                    raise ValueError(f"{file.name}: incomplete connection metadata; cannot safely group flows.")
                left = ep["src_ip"].str.strip() + ":" + ep["src_port"].str.strip()
                right = ep["dst_ip"].str.strip() + ":" + ep["dst_port"].str.strip()
                group = pd.DataFrame({"a": np.minimum(left, right), "b": np.maximum(left, right),
                                      "p": ep["protocol"].str.strip()}, index=ep.index)
                batch["_group"] = pd.util.hash_pandas_object(group, index=False).to_numpy(dtype="uint64")
            batch["_priority"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > max_rows:
                # nsmallest uses a partial selection; memory stays O(K + chunk).
                reservoir = reservoir.nsmallest(max_rows, "_priority")
            if (chunk_index + 1) % 20 == 0:
                log(f"  Scanned {total:,} rows; retained at most {max_rows:,}.")
    if unknown:
        raise ValueError(f"Unrecognized/missing labels: {dict(unknown)}. "
                         "Review the dataset and explicitly extend ids/schema.py; labels are never guessed.")
    if reservoir is None or reservoir.empty:
        raise ValueError("No eligible labeled rows with numeric feature values were found.")
    frame = reservoir.sort_values("_priority").drop(columns="_priority").reset_index(drop=True)
    report = {"source_rows_scanned": total, "embedded_headers_removed": repeated,
              "all_missing_rows_removed": invalid, "eligible_label_counts": dict(sorted(counts.items())),
              "sample_rows": len(frame), "max_rows": max_rows, "seed": seed,
              "sampling": "uniform random priorities across every supplied CSV; independent of label",
              "grouping_available": use_groups, "inventory": inventory}
    return frame, features, report
