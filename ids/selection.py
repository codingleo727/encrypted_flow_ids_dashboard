"""Model-independent selection logic. Call ONLY on 2017 development data."""
from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from .schema import normalize


def pr_auc(y, scores):
    """Trapezoidal PR area, identical for both models; NOT average precision."""
    if len(np.unique(y)) < 2:
        raise ValueError("PR-AUC selection requires both benign and attack examples.")
    precision, recall, _ = precision_recall_curve(y, scores)
    return float(auc(recall, precision))


def optimal_f1_threshold(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
    # Highest threshold breaks ties in favor of fewer flagged flows.
    choices = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(thresholds[choices[-1]])


def rare_label(label):
    s = normalize(label)
    return (s in {"heartbleed", "infiltration", "infilteration", "sqlinjection", "xss", "bot", "botnet"}
            or "webattack" in s or "bruteforce" in s or s in {"ftppatator", "sshpatator"})


def tuning_subset(frame, fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for label, group in frame.groupby("attack_label", sort=True):
        idx = group.index.to_numpy()
        keep = len(idx) if rare_label(label) else max(1, int(np.ceil(len(idx) * fraction)))
        selected.extend(rng.choice(idx, size=keep, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def search_candidates(model, trials, seed=42):
    spaces = {
        "xgboost": {"weight_formula": ["one", "sqrt_ratio", "ratio"],
                    "colsample_bytree": [0.5, 0.7, 0.9], "max_depth": [4, 6, 8],
                    "min_child_weight": [1, 3, 10]},
        "cnn": {"learning_rate": [0.0003, 0.001, 0.003], "dropout": [0.2, 0.35, 0.5],
                "weight_formula": ["one", "sqrt_ratio", "ratio"], "kernel_size": [3, 5, 7]},
    }
    space = spaces[model]
    combos = list(product(*space.values()))
    if not 1 <= trials <= len(combos):
        raise ValueError(f"{model} trials must be between 1 and {len(combos)}.")
    order = np.random.default_rng(seed).permutation(len(combos))[:trials]
    return [dict(zip(space, combos[i])) for i in order]


def positive_weight(y, formula):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if min(counts) == 0:
        raise ValueError("Training requires both classes.")
    ratio = float(counts[0] / counts[1])
    return {"one": 1.0, "sqrt_ratio": ratio ** 0.5, "ratio": ratio}[formula]


def validation_parts(frame, seed=42):
    """50% selection/early-stop, 25% calibration, 25% blend/threshold.

    Preserve the existing connection-group boundaries when metadata exists.
    Returned indices address the supplied validation frame, not source data.
    """
    y = frame["target"].to_numpy(dtype=int)
    if min(np.bincount(y, minlength=2)) < 8:
        raise ValueError("Need at least 8 examples of each class in 2017 validation to separate "
                         "search, calibration and threshold selection; increase max_rows.")
    def split(indices, fraction, offset):
        if "_group" not in frame:
            return train_test_split(indices, test_size=fraction, stratify=y[indices], random_state=seed + offset)
        groups = frame.iloc[indices]["_group"].to_numpy()
        for a, b in GroupShuffleSplit(n_splits=60, test_size=fraction, random_state=seed + offset).split(indices, y[indices], groups):
            if all(min(np.bincount(y[indices[p]], minlength=2)) >= 2 for p in (a, b)):
                return indices[a], indices[b]
        raise ValueError("Not enough independent validation connection groups. Increase max_rows.")
    search, post = split(np.arange(len(frame)), 0.5, 10)
    calibration, decision = split(post, 0.5, 11)
    return {"search": search, "calibration": calibration, "decision": decision}


def logit(scores):
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_sigmoid(y, scores):
    fit = LogisticRegression(C=1.0, solver="lbfgs", random_state=42).fit(logit(scores).reshape(-1, 1), y)
    return {"coefficient": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0])}


def apply_sigmoid(scores, settings):
    if settings is None:
        return np.asarray(scores)
    z = np.clip(settings["coefficient"] * logit(scores) + settings["intercept"], -60, 60)
    return 1 / (1 + np.exp(-z))


def blend_scores(xgb, cnn, settings):
    calibration = settings.get("calibration", {})
    a = apply_sigmoid(xgb, calibration.get("xgboost"))
    b = apply_sigmoid(cnn, calibration.get("cnn"))
    w = settings["xgboost_weight"]
    return w * a + (1 - w) * b


def choose_decisions(y_calibration, calibration_scores, y_decision, decision_scores):
    """Fit calibration on its own subset; choose blend + F1 thresholds elsewhere."""
    def choose(calibration):
        rows = []
        for w in np.linspace(0, 1, 11):
            settings = {"xgboost_weight": float(w), "calibration": calibration}
            score = pr_auc(y_decision, blend_scores(decision_scores["xgboost"], decision_scores["cnn"], settings))
            rows.append({"xgboost_weight": float(w), "pr_auc": score})
        best = max(rows, key=lambda row: (row["pr_auc"], -abs(row["xgboost_weight"] - 0.5)))
        return {"xgboost_weight": best["xgboost_weight"], "calibration": calibration,
                "selection_pr_auc": best["pr_auc"]}, rows
    selected, raw_grid = choose({})
    calibrated_grid = None
    best_base = max(pr_auc(y_decision, s) for s in decision_scores.values())
    if selected["selection_pr_auc"] <= best_base + 1e-12:
        calibration = {name: fit_sigmoid(y_calibration, s) for name, s in calibration_scores.items()}
        candidate, calibrated_grid = choose(calibration)
        if candidate["selection_pr_auc"] > selected["selection_pr_auc"] + 1e-12:
            selected = candidate
    all_scores = {**decision_scores, "ensemble": blend_scores(decision_scores["xgboost"], decision_scores["cnn"], selected)}
    return {"thresholds": {name: optimal_f1_threshold(y_decision, s) for name, s in all_scores.items()},
            "ensemble": selected, "raw_weight_search": raw_grid, "calibrated_weight_search": calibrated_grid,
            "primary_metric": "trapezoidal_pr_auc", "threshold_objective": "binary_attack_f1",
            "source": "2017 validation decision subset; sigmoid fits use separate calibration subset"}
