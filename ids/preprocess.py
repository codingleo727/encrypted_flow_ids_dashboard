"""Train-only numerical transforms; portable NPZ artifacts, no pickle."""
import warnings
import numpy as np
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(medians), medians, 0)
        z = np.log1p(np.where(np.isnan(x), self.median, x))
        self.mean = z.mean(axis=0)
        self.scale = z.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.median):
            raise ValueError("Feature shape does not match saved preprocessing schema.")
        if np.any(np.isinf(x)) or np.any(x < 0):
            raise ValueError("Clean numeric values before preprocessing: nonnegative or NaN expected.")
        z = np.log1p(np.where(np.isnan(x), self.median, x))
        # Bound extreme out-of-distribution standardized values for the CNN.
        return np.clip((z - self.mean) / self.scale, -20, 20).astype("float32")

    def save(self, path):
        np.savez(path, median=self.median, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in ("median", "mean", "scale"):
                setattr(obj, key, data[key])
        return obj


class ModelPreprocessor:
    """Unscaled XGBoost features and median-imputed, RobustScaler CNN inputs.

    All learned values and the feature mask come from full sampled TRAIN only.
    Save numerical arrays rather than pickled sklearn objects.
    """
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        self.active = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if self.active.sum() < 2:
            raise ValueError("Need at least two varying numeric training features.")
        selected = x[:, self.active]
        self.median = np.nanmedian(selected, axis=0)
        filled = np.where(np.isnan(selected), self.median, selected)
        scaler = RobustScaler().fit(filled)
        self.center, self.scale = scaler.center_, scaler.scale_
        self.clip = 100.0
        return self

    def transform(self, x, model="cnn"):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.active):
            raise ValueError("Feature shape does not match saved preprocessing schema.")
        selected = x[:, self.active]
        if np.any(np.isinf(selected)) or np.any(selected < 0):
            raise ValueError("Expected cleaned, nonnegative feature values or NaN.")
        if model == "xgboost":
            return selected.astype("float32")  # native missing-value handling
        if model != "cnn":
            raise ValueError(f"Unknown transform: {model}")
        filled = np.where(np.isnan(selected), self.median, selected)
        return np.clip((filled - self.center) / self.scale, -self.clip, self.clip).astype("float32")

    def save(self, path):
        np.savez(path, active=self.active, median=self.median, center=self.center,
                 scale=self.scale, clip=self.clip)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in ("active", "median", "center", "scale"):
                setattr(obj, key, data[key])
            obj.clip = float(data["clip"])
        return obj
