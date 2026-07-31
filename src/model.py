"""
model.py
=========
Two anomaly-detection approaches, trained per-turbine on that turbine's own
~1 year of (mostly normal) training data — this mirrors how the CARE
benchmark and real predictive-maintenance deployments work: one Normal
Behavior Model (NBM) per asset, not a single global model, because
different turbines/sensors have different baselines.

1. IsolationForestDetector — fast, robust baseline. Good first pass and
   sanity check; scikit-learn only, no GPU needed.
2. AutoencoderDetector — a small dense autoencoder (Keras/TensorFlow).
   Reconstruction error is the anomaly score. This is the approach used
   in the CARE paper's own mini-benchmark and in most published work on
   this dataset, so it's the one to report to your manager as "the model."

Both share a common interface: fit(X_train_normal), score(X) -> anomaly
score per row (higher = more anomalous), and threshold(...) to binarize.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    """Median-impute then standardize. Fit only on normal training data."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Baseline: Isolation Forest
# ---------------------------------------------------------------------------
@dataclass
class IsolationForestDetector:
    contamination: float = config.ASSUMED_CONTAMINATION
    random_state: int = config.RANDOM_STATE

    def __post_init__(self):
        self.preprocessor = make_preprocessor()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame):
        Xp = self.preprocessor.fit_transform(X_train)
        self.model.fit(Xp)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher score = more anomalous (note: sklearn's decision_function
        is higher-for-normal, so we negate it)."""
        Xp = self.preprocessor.transform(X)
        return -self.model.decision_function(Xp)


# ---------------------------------------------------------------------------
# Main approach: Autoencoder-based Normal Behavior Model
# ---------------------------------------------------------------------------
class AutoencoderDetector:
    """
    Dense autoencoder trained to reconstruct normal-operation feature
    vectors. Reconstruction error (MSE) is the anomaly score: the model
    has never seen fault-like patterns, so it reconstructs them poorly.

    Falls back gracefully with a clear error if TensorFlow isn't installed.
    """

    def __init__(self, input_dim: int, encoding_dim: int = None,
                 random_state: int = config.RANDOM_STATE):
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "AutoencoderDetector requires TensorFlow. Install with "
                "`pip install tensorflow --break-system-packages`, or use "
                "IsolationForestDetector instead."
            ) from e

        self._tf = tf
        tf.random.set_seed(random_state)

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or max(8, input_dim // 4)
        self.preprocessor = make_preprocessor()
        self.model = self._build_model()

    def _build_model(self):
        tf = self._tf
        from tensorflow.keras import layers, models

        inp = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(max(self.encoding_dim * 2, 16), activation="relu")(inp)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(self.encoding_dim, activation="relu", name="bottleneck")(x)
        x = layers.Dense(max(self.encoding_dim * 2, 16), activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        out = layers.Dense(self.input_dim, activation="linear")(x)

        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None,
            epochs: int = 50, batch_size: int = 256, verbose: int = 0):
        Xp = self.preprocessor.fit_transform(X_train)
        val_data = None
        callbacks = []
        if X_val is not None and len(X_val) > 0:
            Xv = self.preprocessor.transform(X_val)
            val_data = (Xv, Xv)
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks.append(EarlyStopping(patience=5, restore_best_weights=True))

        self.history = self.model.fit(
            Xp, Xp,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=verbose,
        )
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self.preprocessor.transform(X)
        recon = self.model.predict(Xp, verbose=0)
        return np.mean((Xp - recon) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Thresholding & event post-processing
# ---------------------------------------------------------------------------
def calibrate_threshold(val_scores: np.ndarray,
                         percentile: float = config.THRESHOLD_PERCENTILE) -> float:
    """Set the anomaly threshold from a held-out slice of NORMAL data."""
    return float(np.percentile(val_scores, percentile))


def scores_to_events(scores: np.ndarray, threshold: float,
                      min_event_length: int = config.MIN_EVENT_LENGTH) -> np.ndarray:
    """
    Binarize scores at `threshold`, then suppress isolated flags shorter
    than `min_event_length` consecutive points. This is what keeps a model
    from being penalized (or from raising false alarms) over single-point
    sensor noise — it must sustain anomalous behavior to count.
    """
    flags = (scores >= threshold).astype(int)
    if flags.sum() == 0:
        return flags

    out = flags.copy()
    grp = (flags != np.roll(flags, 1)).cumsum()
    s = pd.Series(flags)
    run_lengths = s.groupby(grp).transform("size")
    out[(flags == 1) & (run_lengths.values < min_event_length)] = 0
    return out
