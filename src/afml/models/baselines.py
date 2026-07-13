"""The three baseline models required by the spec: regression, Random Forest, SVR.

All are sklearn Pipelines (impute → scale → one-hot → model) predicting the
two Paris targets (log10_C, m) jointly.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def make_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    pre = _preprocessor(numeric, categorical)
    return {
        "ridge": Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svr": Pipeline(
            [
                ("pre", pre),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.05))),
            ]
        ),
    }
