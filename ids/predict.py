"""Streaming inference on compatible CSVs; labels are never model inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

from .data import clean_numeric, csv_files
from .inference import PredictionBundle
from .pipeline import manifest_for
from .schema import resolve_columns


def predict_csv(data, run, output, model="both", chunk_size=10000, log=print):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    run = Path(run)
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new output filename.")
    manifest = manifest_for(run)
    features = manifest["features"]
    files = csv_files(data)
    if Path(data).is_dir() and output.resolve().is_relative_to(Path(data).resolve()):
        raise ValueError("Save prediction outputs outside the input dataset folder.")
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Save predictions outside the trained run folder.")
    mappings = []
    for path in files:
        columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig", encoding_errors="replace").columns
        mapping = resolve_columns(columns)
        missing = [f for f in features if f not in mapping]
        if missing:
            raise ValueError(f"{path.name}: missing features: {', '.join(missing)}")
        mappings.append({mapping[f]: f for f in features})
    bundle = PredictionBundle.load(run)
    names = bundle.names if model in ("both", "all") else [model]
    if any(name not in bundle.names for name in names):
        raise ValueError("This run does not contain the requested model.")
    output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    rows = 0
    for path, mapping in zip(files, mappings):
        log(f"Predicting {path.name}...")
        for chunk in pd.read_csv(path, usecols=list(mapping), dtype=str, chunksize=chunk_size,
                                 encoding="utf-8-sig", encoding_errors="replace"):
            numeric = clean_numeric(chunk.rename(columns=mapping), features)
            valid = numeric.notna().any(axis=1).to_numpy()
            source_name = path.resolve().relative_to(Path(data).resolve()).as_posix() if Path(data).is_dir() else path.name
            result = pd.DataFrame({"source_file": source_name, "source_row": chunk.index + 2,
                                   "status": np.where(valid, "ok", "invalid_all_features_missing")})
            raw = numeric.loc[valid].to_numpy(dtype="float32")
            for name in names:
                scores = np.full(len(chunk), np.nan)
                if valid.any():
                    scores[valid] = bundle.predict(raw, name)
                result[f"{name}_score"] = scores
                result[f"{name}_prediction"] = np.where(valid, np.where(scores >= bundle.threshold(name), "Attack", "Benign"), "Invalid")
            result.to_csv(output, index=False, mode="w" if first else "a", header=first,
                          encoding="utf-8-sig" if first else "utf-8")
            first = False
            rows += len(result)
    if first:
        raise ValueError("The supplied CSV files contain no rows to predict.")
    log(f"Wrote {rows:,} rows to {output.resolve()}")
    return rows
