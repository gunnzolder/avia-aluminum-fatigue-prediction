"""Report deliverables: AI-Ready xlsx, predictions/metrics tables, PNG figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afml import viz
from afml.convert import PROCESSED, load

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def export_ai_ready_xlsx(out: Path = REPORTS / "ai_ready_dataset.xlsx", processed: Path = PROCESSED) -> Path:
    """Converted dataset in the AI-Ready multi-sheet structure (spec deliverable)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "materials": load("materials", processed),
        "mechanical_properties": load("mechanical_properties", processed),
        "physical_properties": load("physical_properties", processed),
        "fatigue_data": load("sn_points", processed),
        "DT_Data": load("dt_data", processed),
        "crack_growth_points": load("crack_growth_points", processed),
        "figures_metadata": load("figures", processed),
    }
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return out


def applicability_range(fits: pd.DataFrame, alloy: str) -> tuple[float, float]:
    """ΔK range in which a prediction for `alloy` is considered applicable:
    the median fitted range of training curves from the same alloy series."""
    series = alloy[0]
    pool = fits[fits["alloy"].str.startswith(series)]
    if pool.empty:
        pool = fits
    return float(pool["dK_min"].median()), float(pool["dK_max"].median())


def export_predictions_xlsx(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
    fits: pd.DataFrame,
    out: Path = REPORTS / "predictions.xlsx",
) -> Path:
    """Per target alloy/R/model: C, m, spread band, applicability range."""
    out.parent.mkdir(parents=True, exist_ok=True)
    pred = predictions.copy()
    pred["C_pred"] = 10.0 ** pred["log10_C_pred"]
    ranges = {a: applicability_range(fits, a) for a in pred["alloy"].unique()}
    pred["dK_applicable_min"] = pred["alloy"].map(lambda a: round(ranges[a][0], 2))
    pred["dK_applicable_max"] = pred["alloy"].map(lambda a: round(ranges[a][1], 2))

    wide_iv = intervals.pivot_table(
        index=["alloy", "R_ratio", "model"], columns="target", values=["q05", "q95"]
    )
    wide_iv.columns = [f"{t}_{q}" for q, t in wide_iv.columns]
    pred = pred.merge(wide_iv.reset_index(), on=["alloy", "R_ratio", "model"], how="left")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pred.to_excel(writer, sheet_name="predictions", index=False)
        fits.to_excel(writer, sheet_name="training_fits", index=False)
    return out


def export_figures(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
    fits: pd.DataFrame,
    loao: pd.DataFrame,
    coverage: pd.DataFrame,
    figures_dir: Path = FIGURES,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for alloy, grp in predictions.groupby("alloy"):
        for r_ratio, rgrp in grp.groupby("R_ratio"):
            iv = intervals[(intervals["alloy"] == alloy) & (intervals["R_ratio"] == r_ratio)]
            fig = viz.paris_figure(
                f"Predicted Paris curve — {alloy}, R = {r_ratio}",
                model_params=rgrp,
                intervals=iv,
                dk_range=applicability_range(fits, alloy),
            )
            path = figures_dir / f"paris_{alloy}_R{r_ratio}.png"
            fig.write_image(path, width=900, height=600, scale=2)
            written.append(path)

    for target in ("log10_C", "m"):
        fig = viz.loao_scatter(loao, target)
        path = figures_dir / f"loao_{target}.png"
        fig.write_image(path, width=800, height=600, scale=2)
        written.append(path)

    fig = viz.coverage_heatmap(coverage)
    path = figures_dir / "coverage_matrix.png"
    fig.write_image(path, width=800, height=550, scale=2)
    written.append(path)
    return written
