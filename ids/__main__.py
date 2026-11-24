"""Run with python -m ids; training never accepts an external-test path."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


def doctor():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU architecture: {platform.machine()}, {platform.architecture()[0]}")
    print(f"Logical CPU count: {os.cpu_count()}")
    print("This project runs on CPU; no NVIDIA GPU or NPU is required.")
    names = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
             "xgboost": "xgboost", "torch": "torch", "streamlit": "streamlit"}
    missing = []
    for package, module in names.items():
        if importlib.util.find_spec(module) is None:
            print(f"  MISSING: {package}")
            missing.append(package)
        else:
            print(f"  FOUND: {package}")
    if sys.version_info < (3, 11):
        print("Use Python 3.11 or 3.12 for the documented setup.")
    if missing:
        print("Install dependencies: python -m pip install -r requirements.txt")
    else:
        # Verify native DLLs too, rather than only package metadata.
        import torch
        import xgboost
        torch.set_num_threads(1)
        print(f"Native imports OK: torch {torch.__version__}, xgboost {xgboost.__version__}")
        print("CPU tensor check:", float(torch.tensor([1.0, 2.0]).sum()))
    print("configs/quick.json: 40,000 sampled flows, still the full 18+14 trial budget.")
    print("Run software tests first; configs/smoke.json is for a small synthetic check.")
    return 1 if missing else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Payload-free flow IDS: XGBoost versus tabular 1D-CNN")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Python, architecture and dependencies")
    inspect = commands.add_parser("inspect", help="Inspect CSV headers without training")
    inspect.add_argument("--data", required=True)
    training = commands.add_parser("train", help="Tune, refit and ensemble both models using 2017 ONLY")
    training.add_argument("--data2017", required=True)
    training.add_argument("--run", default="runs/beta")
    training.add_argument("--config", help="JSON overrides for laptop defaults")
    training.add_argument("--max-rows", type=int)
    training.add_argument("--epochs", type=int)
    training.add_argument("--threads", type=int)
    training.add_argument("--xgb-trials", type=int)
    training.add_argument("--cnn-trials", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate saved models on unseen 2018 CSVs")
    evaluate.add_argument("--data2018", required=True)
    evaluate.add_argument("--run", default="runs/beta")
    evaluate.add_argument("--max-rows", type=int, default=150000)
    evaluate.add_argument("--chunk-size", type=int, default=25000)
    prediction = commands.add_parser("predict", help="Predict a compatible CSV or folder, with optional labels")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--run", default="runs/beta")
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--model", choices=["all", "both", "xgboost", "cnn", "ensemble"], default="all")
    prediction.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "inspect":
            from .data import inspect_files
            inventory = inspect_files(args.data)
            for item in inventory:
                print(f"{Path(item['path']).name}: {len(item['features'])} supported features, {item['bytes']/1e6:.1f} MB")
                print("  " + ", ".join(item["features"]))
            return 0
        if args.command == "train":
            from .pipeline import DEFAULT_CONFIG, train
            config = DEFAULT_CONFIG.copy()
            if args.config:
                supplied = json.loads(Path(args.config).read_text(encoding="utf-8"))
                if not isinstance(supplied, dict):
                    raise ValueError("Config must be a JSON object.")
                unknown = supplied.keys() - config.keys()
                if unknown:
                    raise ValueError(f"Unknown config settings: {sorted(unknown)}")
                config.update(supplied)
            for key in ("max_rows", "epochs", "threads", "xgb_trials", "cnn_trials"):
                if getattr(args, key) is not None:
                    config[key] = getattr(args, key)
            train(args.data2017, args.run, config)
        elif args.command == "evaluate":
            from .pipeline import evaluate_external
            evaluate_external(args.data2018, args.run, args.max_rows, args.chunk_size)
        else:
            from .predict import predict_csv
            predict_csv(args.data, args.run, args.output, args.model, args.chunk_size)
        return 0
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
