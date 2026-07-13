"""Validation schemes required by the spec: random, group, and leave-one-alloy-out.

The unit of observation is one fitted curve. Group split by dataset_id guards
against leakage between figures digitized from the same physical experiment;
LOAO tests prediction for an alloy entirely absent from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from afml.features import TARGETS

RANDOM_STATE = 42


def _splitters(df: pd.DataFrame):
    n_datasets = df["dataset_id"].nunique()
    return {
        "random": (KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), None),
        "group_dataset": (GroupKFold(n_splits=min(5, n_datasets)), df["dataset_id"]),
        "leave_one_alloy_out": (LeaveOneGroupOut(), df["alloy"]),
    }


def evaluate_models(models: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Cross-validate every model under every scheme; tidy metrics per target."""
    X, y = df[feature_cols], df[TARGETS]
    rows = []
    for scheme, (splitter, groups) in _splitters(df).items():
        for name, model in models.items():
            preds = pd.DataFrame(index=df.index, columns=TARGETS, dtype=float)
            held_alloy = pd.Series(index=df.index, dtype="string")
            for train_idx, test_idx in splitter.split(X, y, groups):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
                held_alloy.iloc[test_idx] = df["alloy"].iloc[test_idx]
            for target in TARGETS:
                rows.append(
                    {
                        "scheme": scheme,
                        "model": name,
                        "target": target,
                        "r2": r2_score(y[target], preds[target]),
                        "mae": mean_absolute_error(y[target], preds[target]),
                        "rmse": root_mean_squared_error(y[target], preds[target]),
                    }
                )
    return pd.DataFrame(rows)


def loao_predictions(models: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Out-of-fold LOAO predictions per curve — the spec's known-alloy check."""
    X, y = df[feature_cols], df[TARGETS]
    frames = []
    for name, model in models.items():
        preds = pd.DataFrame(index=df.index, columns=TARGETS, dtype=float)
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, df["alloy"]):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
        out = df[["curve_id", "alloy", "R_ratio"]].copy()
        out["model"] = name
        for target in TARGETS:
            out[f"{target}_true"] = y[target].to_numpy()
            out[f"{target}_pred"] = preds[target].to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def fit_full_and_predict(
    models: dict, df: pd.DataFrame, feature_cols: list[str], X_new: pd.DataFrame
) -> pd.DataFrame:
    """Fit each model on all training curves and predict targets for new rows."""
    rows = []
    for name, model in models.items():
        model.fit(df[feature_cols], df[TARGETS])
        pred = model.predict(X_new[feature_cols])
        for i, (_, feat_row) in enumerate(X_new.iterrows()):
            rows.append(
                {
                    "alloy": feat_row["alloy"],
                    "R_ratio": feat_row["R_ratio"],
                    "model": name,
                    "log10_C_pred": float(pred[i][0]),
                    "m_pred": float(pred[i][1]),
                }
            )
    return pd.DataFrame(rows)
