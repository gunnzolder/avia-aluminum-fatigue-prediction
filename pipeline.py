"""End-to-end pipeline CLI.

Usage:
    uv run python pipeline.py convert   # xlsx → validated Parquet + quality report
    uv run python pipeline.py fit       # Paris fits + crosscheck + physics audit
    uv run python pipeline.py train     # baselines, split metrics, LOAO
    uv run python pipeline.py predict   # target-alloy predictions + intervals
    uv run python pipeline.py export    # xlsx/csv deliverables + PNG figures
    uv run python pipeline.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import yaml

# Avoid pyarrow's bundled mimalloc: it segfaults intermittently on macOS arm64
# in threaded use (same fix as app/dashboard.py).
pa.set_memory_pool(pa.system_memory_pool())

from afml import export, physics
from afml.convert import PROCESSED, convert_all, load
from afml.coverage import coverage_matrix
from afml.features import (
    alloy_property_table,
    build_features,
    feature_columns,
    target_feature_rows,
)
from afml.models.baselines import make_models
from afml.models.paris_fit import crosscheck_reference, fit_all_curves
from afml.models.splits import evaluate_models, fit_full_and_predict, loao_predictions
from afml.models.uncertainty import prediction_intervals

REPORTS = Path("reports")


def _training_frame() -> tuple[pd.DataFrame, list[str]]:
    fits = load("paris_fits")
    train = fits[fits["train_ok"]].reset_index(drop=True)
    props = alloy_property_table(
        load("materials"), load("mechanical_properties"), load("physical_properties")
    )
    feat = build_features(train, props)
    numeric, categorical = feature_columns(feat)
    return feat, numeric + categorical


def step_convert() -> None:
    REPORTS.mkdir(exist_ok=True)
    reports = convert_all()
    lines = []
    for r in reports:
        lines.append(f"{r.table}: {r.n_rows} rows, {r.n_dropped_duplicates} duplicates dropped")
        for col, n in r.n_missing_by_col.items():
            lines.append(f"  missing {col}: {n}")
        lines += [f"  NOTE: {n}" for n in r.notes]
    (REPORTS / "data_quality.txt").write_text("\n".join(lines))
    coverage_matrix().to_csv(REPORTS / "coverage_matrix.csv", index=False)
    print("\n".join(lines))
    print(f"\nwrote {PROCESSED}/*.parquet, reports/coverage_matrix.csv, reports/data_quality.txt")


def step_fit() -> None:
    points = load("crack_growth_points")
    fits = fit_all_curves(points)

    point_audit = physics.audit_curve_points(points)
    fit_audit = physics.audit_fit(fits)
    audit = pd.concat([point_audit, fit_audit], ignore_index=True)
    audit.to_parquet(PROCESSED / "physics_audit.parquet")

    fits["train_ok"] = physics.training_mask(fits, point_audit)
    fits.to_parquet(PROCESSED / "paris_fits.parquet")

    cc = crosscheck_reference(fits, load("reference_fits"))
    cc.to_csv(REPORTS / "fit_crosscheck.csv", index=False)

    n_fail = (audit["level"] == "FAIL").sum()
    n_warn = (audit["level"] == "WARN").sum()
    print(f"fitted {len(fits)} curves; training-eligible {int(fits['train_ok'].sum())}")
    print(f"physics audit: {n_fail} FAIL, {n_warn} WARN → data/processed/physics_audit.parquet")
    print(f"crosscheck vs published fits: {len(cc)} rows → reports/fit_crosscheck.csv")


def step_train() -> None:
    feat, cols = _training_frame()
    numeric, categorical = feature_columns(feat)
    models = make_models(numeric, categorical)

    metrics = evaluate_models(models, feat, cols)
    metrics.to_csv(REPORTS / "metrics.csv", index=False)
    loao = loao_predictions(models, feat, cols)
    loao.to_parquet(PROCESSED / "loao_predictions.parquet")

    print(metrics.round(3).to_string(index=False))
    print(f"\nwrote reports/metrics.csv, {PROCESSED}/loao_predictions.parquet")


def step_predict() -> None:
    cfg = yaml.safe_load(Path("targets.yaml").read_text())
    feat, cols = _training_frame()
    numeric, categorical = feature_columns(feat)
    props = alloy_property_table(
        load("materials"), load("mechanical_properties"), load("physical_properties")
    )

    x_new = pd.concat(
        [
            target_feature_rows(alloy, props, cfg["predict_at_R"])
            for alloy in cfg["target_alloys"]
        ],
        ignore_index=True,
    )
    models = make_models(numeric, categorical)
    predictions = fit_full_and_predict(models, feat, cols, x_new)
    intervals = prediction_intervals(models, feat, cols, x_new)

    pred_audit = physics.audit_predictions(predictions)
    predictions.to_parquet(PROCESSED / "predictions.parquet")
    intervals.to_parquet(PROCESSED / "prediction_intervals.parquet")
    pred_audit.to_parquet(PROCESSED / "prediction_audit.parquet")

    print(predictions.round(3).to_string(index=False))
    print(f"\nprediction audit: {(pred_audit['level'] == 'FAIL').sum()} FAIL, "
          f"{(pred_audit['level'] == 'WARN').sum()} WARN")


def step_export() -> None:
    fits = load("paris_fits")
    predictions = load("predictions")
    intervals = load("prediction_intervals")
    loao = load("loao_predictions")
    coverage = pd.read_csv(REPORTS / "coverage_matrix.csv", dtype={"alloy": str})

    p1 = export.export_ai_ready_xlsx()
    p2 = export.export_predictions_xlsx(predictions, intervals, fits)
    figs = export.export_figures(predictions, intervals, fits, loao, coverage)
    print(f"wrote {p1}, {p2}, and {len(figs)} figures in {export.FIGURES}/")


STEPS = {"convert": step_convert, "fit": step_fit, "train": step_train,
         "predict": step_predict, "export": step_export}


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for name, fn in STEPS.items():
            print(f"\n=== {name} ===")
            fn()
    elif arg in STEPS:
        STEPS[arg]()
    else:
        sys.exit(f"unknown step {arg!r}; use one of {list(STEPS)} or 'all'")


if __name__ == "__main__":
    main()
