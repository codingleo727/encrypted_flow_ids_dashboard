"""Binary metrics with explicit denominators and batched CPU latency."""
from time import perf_counter
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from .selection import pr_auc, rare_label


def binary_metrics(y, scores, threshold=0.5):
    y = np.asarray(y, dtype="int8")
    scores = np.asarray(scores)
    predicted = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"n": len(y), "benign": int(tn + fp), "attacks": int(tp + fn),
            "threshold": threshold, "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)) if tp + fn else None,
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predicted, labels=[0, 1], average="weighted", zero_division=0)),
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both else None,
            "average_precision": float(average_precision_score(y, scores)) if both else None,
            "pr_auc": pr_auc(y, scores) if both else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def benchmark(predict, x, batch_size=1024, repeats=3):
    """Median warmed model-only time; feature transforms and loading excluded."""
    x = x[:min(len(x), 10000)]
    predict(x[:min(batch_size, len(x))])
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        for i in range(0, len(x), batch_size):
            predict(x[i:i + batch_size])
        durations.append(perf_counter() - start)
    elapsed = float(np.median(durations))
    return {"model_us_per_flow": elapsed * 1e6 / len(x), "benchmark_rows": len(x),
            "benchmark_batch_size": batch_size, "benchmark_repeats": repeats,
            "latency_scope": "warmed CPU model-only batched prediction; median of repetitions"}


def per_attack_metrics(frame, scores, threshold=0.5):
    output = []
    for label in sorted(frame["attack_label"].unique()):
        mask = frame["attack_label"].eq(label).to_numpy()
        y = frame.loc[mask, "target"].to_numpy()
        pred = scores[mask] >= threshold
        output.append({"attack_label": label, "n": int(mask.sum()),
                       "target": int(y[0]), "rare_attack": bool(rare_label(label)),
                       "recall": float(pred.mean()) if y[0] else None,
                       "false_positive_rate": float(pred.mean()) if not y[0] else None,
                       "predicted_attack_rate": float(pred.mean())})
    return output
