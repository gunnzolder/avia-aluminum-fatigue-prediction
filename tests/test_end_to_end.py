"""End-to-end smoke tests against the real source workbooks and pipeline outputs.

These run the real converters (fast, <10 s) and assert invariants of the actual
data. Tests that need model outputs are skipped until `pipeline.py all` has run.
"""

from pathlib import Path

import pandas as pd
import pytest

from afml.convert.chart_meta import figures_table
from afml.convert.crack_growth import crack_growth_points

RAW = Path("inbox/Fatigue Crack Growth Database.xlsx")
PROCESSED = Path("data/processed")


@pytest.fixture(scope="module")
def figures():
    return figures_table(str(RAW))


def test_all_figures_have_metadata(figures):
    assert len(figures.figures) == 49
    assert figures.figures["alloy"].notna().all()
    assert not figures.unparsed_titles


def test_points_parse_from_every_data_sheet(figures):
    pts = crack_growth_points(str(RAW), figures)
    assert pts["page"].nunique() == 49
    assert len(pts) > 2500
    assert set(pts["alloy"].unique()) == {"2014", "2024", "7050", "7075", "7475"}


def test_negative_figures_contain_negative_R(figures):
    # Every NEGATIVE-marked figure must yield at least one R < 0 curve. Positive R
    # can legitimately coexist on sign-aware sheets (explicit minus in labels).
    pts = crack_growth_points(str(RAW), figures)
    merged = pts.merge(figures.figures[["page", "r_sign"]], on="page")
    neg_pages = merged[merged["r_sign"] == "NEGATIVE"].groupby("page")["R_ratio"].min()
    assert (neg_pages < 0).all(), neg_pages.to_string()


needs_pipeline = pytest.mark.skipif(
    not (PROCESSED / "predictions.parquet").exists(),
    reason="run `uv run python pipeline.py all` first",
)


@needs_pipeline
def test_pipeline_outputs_exist():
    for table in [
        "crack_growth_points", "paris_fits", "physics_audit",
        "loao_predictions", "predictions", "prediction_intervals",
    ]:
        assert (PROCESSED / f"{table}.parquet").exists(), table
    for report in ["coverage_matrix.csv", "metrics.csv", "predictions.xlsx", "ai_ready_dataset.xlsx"]:
        assert Path("reports", report).exists(), report


@needs_pipeline
def test_predictions_cover_all_targets_and_models():
    import yaml

    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    cfg = yaml.safe_load(Path("targets.yaml").read_text())
    assert set(preds["alloy"]) == set(cfg["target_alloys"])
    assert set(preds["model"]) == {"ridge", "random_forest", "svr"}
    assert len(preds) == len(cfg["target_alloys"]) * len(cfg["predict_at_R"]) * 3


@needs_pipeline
def test_no_unexplained_prediction_failures():
    audit = pd.read_parquet(PROCESSED / "prediction_audit.parquet")
    fails = audit[audit["level"] == "FAIL"]
    assert fails.empty, fails.to_string()


@needs_pipeline
def test_dashboard_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/dashboard.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
