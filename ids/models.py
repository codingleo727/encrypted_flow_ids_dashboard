"""CPU model fitting, with explicit PR-AUC maximization and portable artifacts."""
import copy
import random
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from .selection import pr_auc, positive_weight


class FlowCNN(nn.Module):
    """Convolution over related aggregate-feature positions, not packet time."""
    def __init__(self, n_features, kernel_size=3, dropout=0.35, channels=(32, 64, 128)):
        super().__init__()
        self.n_features = n_features
        self.architecture = {"kernel_size": kernel_size, "dropout": dropout, "channels": list(channels)}
        layers = []
        before = 1
        for width in channels:
            layers.extend([nn.Conv1d(before, width, kernel_size, padding=kernel_size // 2),
                           nn.BatchNorm1d(width), nn.ReLU()])
            before = width
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(before, 64),
                       nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.unsqueeze(1)).squeeze(1)


class LegacyFlowCNN(nn.Module):
    """Load original beta runs without silently changing their architecture."""
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(), nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return self.network(x.unsqueeze(1)).squeeze(1)


def configure_cpu(threads, seed):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def cnn_scores(model, x, batch_size=512):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            inputs = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size], dtype="float32"))
            outputs.append(torch.sigmoid(model(inputs)).numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype="float32")


def fit_xgb(x_train, y_train, x_val, y_val, params, config, log=print):
    settings = dict(device="cpu", tree_method="hist", n_estimators=config["trees"],
                    learning_rate=0.05, subsample=0.85, colsample_bylevel=0.8,
                    gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                    n_jobs=config["threads"], random_state=config["seed"])
    settings.update({k: v for k, v in params.items() if k != "weight_formula"})
    settings["scale_pos_weight"] = positive_weight(y_train, params["weight_formula"])
    # Unlike native 'aucpr', this callable matches the exact area used to rank
    # CNNs and ensembles. Explicit maximize avoids callable-cost default behavior.
    model = XGBClassifier(**settings, eval_metric=pr_auc, callbacks=[EarlyStopping(
        rounds=config["xgb_patience"], metric_name="pr_auc", data_name="validation_0",
        maximize=True, save_best=True)])
    start = perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    history = [float(v) for v in model.evals_result()["validation_0"]["pr_auc"]]
    return model, {"seconds": perf_counter() - start, "history": history,
                   "best_iteration": int(model.best_iteration), "positive_weight": settings["scale_pos_weight"]}


class EpochBudgetExceeded(RuntimeError):
    pass


def fit_cnn(x_train, y_train, x_val, y_val, params, config, log=print, enforce_budget=True):
    configure_cpu(config["threads"], config["seed"])
    channels = params.get("channels", [32, 64, 128])
    batch_size = params.get("batch_size", config["batch_size"])
    model = FlowCNN(x_train.shape[1], params["kernel_size"], params["dropout"], channels)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype("float32"))),
                        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    weight = positive_weight(y_train, params["weight_formula"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    best_score, best_epoch, best_state, wait = -1.0, 0, None, 0
    history = []
    start = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = perf_counter()
        model.train()
        total = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CNN loss; inspect inputs and RobustScaler outputs.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.item()) * len(labels)
        scores = cnn_scores(model, x_val, batch_size)
        area = pr_auc(y_val, scores)
        # Report ordinary validation log loss separately from weighted train loss.
        p = np.clip(scores.astype(float), 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
        elapsed = perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": total / len(x_train), "validation_loss": val_loss,
                        "validation_pr_auc": area, "epoch_seconds": elapsed,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        log(f"    epoch {epoch:02d}: validation PR-AUC={area:.5f}, {elapsed:.1f}s")
        if area > best_score + 1e-8:
            best_score, best_epoch, best_state, wait = area, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        scheduler.step(area)
        if enforce_budget and elapsed > config["cnn_epoch_budget_seconds"]:
            raise EpochBudgetExceeded(f"Epoch took {elapsed:.1f}s, exceeding {config['cnn_epoch_budget_seconds']}s. "
                                      "Remaining trials will use smaller channels and batches of 1024.")
        if wait >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seconds": perf_counter() - start, "history": history, "best_epoch": best_epoch,
                   "positive_weight": weight, "channels": list(channels), "batch_size": batch_size}


def save_cnn(model, path):
    torch.save({"n_features": model.n_features, "architecture": model.architecture,
                "state_dict": model.state_dict()}, path)


def save_models(models, directory, n_features=None):
    models["xgboost"].save_model(directory / "xgboost.json")
    save_cnn(models["cnn"], directory / "cnn.pt")


def load_model(directory, name, threads=4):
    if name == "xgboost":
        model = XGBClassifier(n_jobs=threads, device="cpu")
        model.load_model(directory / "xgboost.json")
        model.set_params(n_jobs=threads, device="cpu")
        return model
    if name != "cnn":
        raise ValueError(f"Unknown model: {name}")
    torch.set_num_threads(threads)
    saved = torch.load(directory / "cnn.pt", map_location="cpu", weights_only=True)
    model = (FlowCNN(saved["n_features"], **saved["architecture"]) if "architecture" in saved
             else LegacyFlowCNN(saved["n_features"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def score_model(model, name, x):
    return model.predict_proba(x)[:, 1] if name == "xgboost" else cnn_scores(model, x)
