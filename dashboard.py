"""Streamlit demo: saved evaluation results plus fresh sample predictions."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st

from ids.metrics import binary_metrics
from ids.inference import PredictionBundle
from ids.pipeline import manifest_for

st.set_page_config(page_title="Encrypted Flow IDS", page_icon="🛡️", layout="wide")
st.title("Encrypted Flow IDS")
st.caption("Offline network flow classification · XGBoost, 1D-CNN and Ensemble · Benign / Attack")

ROOT = Path(__file__).resolve().parent
run_text = st.sidebar.text_input("Experiment folder", value="runs/beta")
run = Path(run_text).expanduser()
if not run.is_absolute():
    run = ROOT / run
if not (run / "manifest.json").is_file():
    st.info("Train the models first, then open this experiment folder.")
    st.code("python -m ids train --data2017 data/raw/2017 --run runs/beta --config configs/quick.json")
    st.stop()
try:
    manifest = manifest_for(run)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

if manifest.get("source_kind") == "synthetic_fixture":
    st.warning("SYNTHETIC SOFTWARE TEST DATA — these numbers are not CIC benchmark results.")

datasets = {"2017 held-out test": ("2017_test", "demo_2017.npz")}
if (run / "demo_2018.npz").is_file() and (run / "2018_unseen" / "metrics.json").is_file():
    datasets["2018 unseen evaluation"] = ("2018_unseen", "demo_2018.npz")
else:
    st.sidebar.info("Add 2018 results with the evaluate command in QUICKSTART.txt.")
selection = st.sidebar.selectbox("Dataset", list(datasets))
dataset, demo_name = datasets[selection]
labels = {"XGBoost": "xgboost", "1D-CNN": "cnn"}
if manifest.get("schema_version", 1) >= 2:
    labels["Ensemble"] = "ensemble"
model_name = st.sidebar.selectbox("Model", list(labels))
key = labels[model_name]

# Key caches by artifact modification times to avoid stale runs after training.
signature = tuple((p.name, p.stat().st_mtime_ns) for p in sorted(run.rglob("*")) if p.is_file())

@st.cache_resource
def resources(run_path, file_signature):
    directory = Path(run_path)
    return PredictionBundle.load(directory)

@st.cache_data
def load_demo(path, file_signature):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

bundle = resources(str(run), signature)
threshold = bundle.threshold(key)
st.sidebar.caption(f"Decision threshold: {threshold:.6g}; selected on 2017 validation for new runs.")
demo = load_demo(str(run / demo_name), signature)
recorded = json.loads((run / dataset / "metrics.json").read_text(encoding="utf-8"))
tabs = st.tabs(["Run detection", "Model comparison", "Experiment details"])

def percent(value):
    return "N/A" if value is None else f"{value:.2%}"

with tabs[0]:
    st.subheader("Run predictions")
    st.write(f"{selection}: {len(demo['y']):,} saved sample flows available for live inference.")
    sample_options = sorted(set([min(100, len(demo['y'])), min(500, len(demo['y'])), len(demo['y'])]))
    size = (st.select_slider("Flows to analyze", options=sample_options, value=len(demo["y"]))
            if len(sample_options) > 1 else sample_options[0])
    state_key = (str(run), signature, dataset, key, size)
    if st.button("Run detection", type="primary"):
        with st.spinner("Classifying flows on CPU..."):
            started = perf_counter()
            scores = bundle.predict(demo["x"][:size], key)
            elapsed = perf_counter() - started
        # Labels enter only metric calculation, never the prediction inputs.
        st.session_state["detection"] = {"key": state_key, "scores": scores, "seconds": elapsed}
    result = st.session_state.get("detection")
    if result and result["key"] == state_key:
        scores = result["scores"]
        pred = scores >= threshold
        y = demo["y"][:size]
        metrics = binary_metrics(y, scores, threshold)
        a, b, c, d = st.columns(4)
        a.metric("Flows analyzed", f"{size:,}")
        b.metric("Flagged as attack", f"{int(pred.sum()):,}")
        c.metric("Sample F1", f"{metrics['f1']:.3f}")
        d.metric("Inference + preprocessing", f"{result['seconds']*1000:.1f} ms")
        st.caption("Metrics here describe this demo sample. Model comparison uses the entire saved evaluation sample.")
        left, right = st.columns([1, 2])
        with left:
            st.bar_chart(pd.DataFrame({"Flows": [int((~pred).sum()), int(pred.sum())]}, index=["Benign", "Attack"]))
        with right:
            st.write("Confusion matrix (rows = actual, columns = predicted)")
            st.dataframe(pd.DataFrame([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                                     index=["Actual benign", "Actual attack"],
                                     columns=["Predicted benign", "Predicted attack"]), width="stretch")
        table = pd.DataFrame({"Source file": demo["source_file"][:size], "CSV row": demo["source_row"][:size],
                              "Actual label": demo["attack_label"][:size],
                              "Prediction": np.where(pred, "Attack", "Benign"),
                              "Attack score": scores})
        only_flagged = st.checkbox("Show flagged flows only", value=False)
        shown = table.loc[pred] if only_flagged else table
        st.dataframe(shown, hide_index=True, width="stretch")
        st.download_button("Download predictions", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{dataset}_{key}_demo.csv", mime="text/csv")
    else:
        st.info("Select a dataset and model, then run detection.")

with tabs[1]:
    st.subheader(f"Recorded results: {selection}")
    st.caption(f"Evaluated flows: {recorded['xgboost']['n']:,}. These are measured results for this experiment.")
    comparison = pd.DataFrame.from_dict(recorded, orient="index")
    timing_key = "inference_us_per_flow" if "inference_us_per_flow" in comparison else "model_us_per_flow"
    comparison["latency"] = comparison[timing_key]
    metric_columns = [c for c in ["n", "threshold", "pr_auc", "average_precision", "precision", "recall",
                                  "f1", "macro_f1", "weighted_f1", "fpr", "latency"] if c in comparison]
    st.dataframe(comparison[metric_columns].rename(columns={"latency": "CPU inference µs/flow"}), width="stretch")
    st.caption("PR-AUC uses trapezoidal integration for every model; average precision is reported separately.")
    st.bar_chart(comparison[["precision", "recall", "f1"]])
    a, b = st.columns(2)
    with a:
        st.write("False-positive rate")
        st.bar_chart(comparison[["fpr"]])
    with b:
        st.write("CPU inference latency (µs/flow)")
        st.bar_chart(comparison[["latency"]])
    st.caption("New runs: warmed batched CPU inference including preprocessing and ensemble work; "
               "median of 3 repetitions over up to 10,000 flows. Excludes CSV I/O and model loading. "
               "Original beta runs measured model-only latency. This is not single-flow response time.")
    st.write("Detection by original attack label")
    st.dataframe(pd.read_csv(run / dataset / "attack_breakdown.csv"), hide_index=True, width="stretch")
    st.caption("Predicted attack rate is recall for attack labels and false-positive rate for benign labels.")
    if (run / "2018_unseen" / "metrics.json").is_file():
        st.write("Generalization: 2017 → 2018")
        m17 = json.loads((run / "2017_test" / "metrics.json").read_text())
        m18 = json.loads((run / "2018_unseen" / "metrics.json").read_text())
        rows = [{"model": name, "2017 F1": m17[name]["f1"], "2018 F1": m18[name]["f1"],
                 "F1 change": m18[name]["f1"] - m17[name]["f1"]} for name in m17]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tabs[2]:
    st.write("Both models use the same selected aggregate features and data split. "
             "New runs use unscaled XGBoost inputs and training-fitted RobustScaler inputs for the CNN.")
    if (run / "decisions.json").is_file():
        st.write("Ensemble and threshold selection")
        settings = json.loads((run / "decisions.json").read_text(encoding="utf-8"))
        st.json({"thresholds": settings["thresholds"], "ensemble": settings["ensemble"]})
        st.caption("Ensemble sigmoid calibration is used only if selected on 2017 validation. "
                   "Scores are not guaranteed calibrated probabilities.")
        trials = pd.read_csv(run / "trials.csv")
        st.write("Hyperparameter trials")
        st.dataframe(trials, hide_index=True, width="stretch")
    st.write("The CNN convolves over fixed feature positions. CSV summaries do not contain packet ordering.")
    st.write("The system uses no packet payloads. This experiment does not establish that all benchmark flows are encrypted.")
    st.write("2018 is reserved for external evaluation. Use 2017 validation data for model selection.")
    st.json(json.loads((run / "split.json").read_text(encoding="utf-8")))
    st.write("Feature order")
    st.code("\n".join(manifest["features"]))
    st.write("Training loss")
    history = pd.read_csv(run / "cnn_history.csv").set_index("epoch")
    st.line_chart(history[["train_loss", "validation_loss"]])
    st.caption("Training loss is class weighted; validation log loss is unweighted for new runs.")
    if "validation_pr_auc" in history:
        st.line_chart(history[["validation_pr_auc"]])
    st.write("XGBoost feature importance")
    importance = pd.read_csv(run / "feature_importance.csv").head(12).set_index("feature")
    st.bar_chart(importance)
