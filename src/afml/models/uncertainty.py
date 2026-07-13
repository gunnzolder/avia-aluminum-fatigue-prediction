"""Prediction-interval estimation.

Random Forest: spread of per-tree predictions. Ridge/SVR: bootstrap refits on
resampled training curves. Both yield (q05, q50, q95) per target — a spread
band, not a calibrated confidence interval (research-grade, per spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from afml.features import TARGETS

N_BOOTSTRAP = 200
RANDOM_STATE = 42
QUANTILES = (0.05, 0.5, 0.95)


def rf_interval(fitted_rf_pipeline, X_new: pd.DataFrame) -> np.ndarray:
    """Quantiles over per-tree predictions: (n_rows, n_targets, n_quantiles)."""
    pre = fitted_rf_pipeline.named_steps["pre"]
    forest = fitted_rf_pipeline.named_steps["model"]
    Xt = pre.transform(X_new)
    per_tree = np.stack([tree.predict(Xt) for tree in forest.estimators_])
    return np.quantile(per_tree, QUANTILES, axis=0).transpose(1, 2, 0)


def bootstrap_interval(
    model_pipeline, df: pd.DataFrame, feature_cols: list[str], X_new: pd.DataFrame
) -> np.ndarray:
    """Bootstrap-refit quantiles: (n_rows, n_targets, n_quantiles)."""
    rng = np.random.default_rng(RANDOM_STATE)
    preds = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(df), size=len(df), replace=True)
        sample = df.iloc[idx]
        m = clone(model_pipeline)
        m.fit(sample[feature_cols], sample[TARGETS])
        preds.append(m.predict(X_new[feature_cols]))
    stacked = np.stack(preds)
    return np.quantile(stacked, QUANTILES, axis=0).transpose(1, 2, 0)


def prediction_intervals(
    models: dict, df: pd.DataFrame, feature_cols: list[str], X_new: pd.DataFrame
) -> pd.DataFrame:
    """Tidy intervals for every model/new-row/target."""
    rows = []
    for name, pipeline in models.items():
        pipeline.fit(df[feature_cols], df[TARGETS])
        if name == "random_forest":
            q = rf_interval(pipeline, X_new)
        else:
            q = bootstrap_interval(pipeline, df, feature_cols, X_new)
        for i, (_, feat_row) in enumerate(X_new.iterrows()):
            for t, target in enumerate(TARGETS):
                rows.append(
                    {
                        "alloy": feat_row["alloy"],
                        "R_ratio": feat_row["R_ratio"],
                        "model": name,
                        "target": target,
                        "q05": float(q[i, t, 0]),
                        "q50": float(q[i, t, 1]),
                        "q95": float(q[i, t, 2]),
                    }
                )
    return pd.DataFrame(rows)
