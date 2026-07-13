"""Streamlit dashboard for the aluminum-fatigue ML pipeline.

Run:  uv run streamlit run app/dashboard.py
Requires pipeline outputs:  uv run python pipeline.py all
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st

# pyarrow's bundled mimalloc segfaults on macOS arm64 when Streamlit serializes
# DataFrames on worker threads (mi_thread_init race, see crash reports). Route
# all Arrow allocations through the OS allocator instead.
pa.set_memory_pool(pa.system_memory_pool())

try:
    from afml import viz
except ImportError:  # hosted platforms may install deps without the project itself
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from afml import viz

from afml.convert import PROCESSED, load
from afml.export import applicability_range

st.set_page_config(page_title="Aluminum Fatigue ML", layout="wide")


def _bootstrap_if_needed() -> None:
    """First run on a fresh clone (e.g. Streamlit Community Cloud): build the
    dataset, models, and report tables from the source workbooks in inbox/."""
    if (PROCESSED / "predictions.parquet").exists():
        return
    from afml import pipeline_steps

    with st.status("First run: building dataset and models (~2–5 min)…", expanded=True) as status:
        for label, fn in [
            ("Converting source workbooks", pipeline_steps.step_convert),
            ("Fitting Paris curves + physics audit", pipeline_steps.step_fit),
            ("Training models (3 split schemes)", pipeline_steps.step_train),
            ("Predicting target alloys + uncertainty", pipeline_steps.step_predict),
        ]:
            st.write(label)
            fn()
        st.write("Exporting report tables")
        try:
            pipeline_steps.step_export(include_figures=True)
        except Exception:
            pipeline_steps.step_export(include_figures=False)
            st.write("PNG figure export skipped (no Chromium runtime for kaleido)")
        status.update(label="Dataset and models ready", state="complete", expanded=False)
    st.rerun()


_bootstrap_if_needed()


@st.cache_data
def _load(table: str) -> pd.DataFrame:
    return load(table)


def _require(table: str) -> pd.DataFrame | None:
    if not (PROCESSED / f"{table}.parquet").exists():
        st.warning(f"`{table}` not found — run `uv run python pipeline.py all` first.")
        return None
    return _load(table)


st.title("Aluminum Fatigue ML — Paris curve prediction")

tab_cov, tab_curves, tab_models, tab_audit, tab_downloads = st.tabs(
    ["Coverage", "Curves", "Models", "Physics audit", "Downloads"]
)

with tab_cov:
    cov_path = Path("reports/coverage_matrix.csv")
    if not cov_path.exists():
        st.warning("Run `uv run python pipeline.py convert` first.")
    else:
        cov = pd.read_csv(cov_path, dtype={"alloy": str})
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(viz.coverage_heatmap(cov), width="stretch")
        with right:
            st.subheader("Coverage table")
            st.dataframe(cov, hide_index=True, height=420)
            targets = cov[(cov.n_static_props > 0) & (cov.n_cg_curves == 0)]["alloy"]
            st.markdown(
                "**Prediction targets** (static props, no da/dN curves): "
                + ", ".join(targets)
            )

with tab_curves:
    points = _require("crack_growth_points")
    fits = _require("paris_fits")
    if points is not None and fits is not None:
        ref = _load("reference_fits")
        c1, c2, c3 = st.columns(3)
        alloy = c1.selectbox("Alloy", sorted(points["alloy"].unique()))
        sub = points[points["alloy"] == alloy]
        r_options = ["all"] + sorted(sub["R_ratio"].unique().tolist())
        r_sel = c2.selectbox("R ratio", r_options)
        fit_sel = c3.selectbox("Source figure fit type", ["all"] + sorted(sub["fit_type"].dropna().unique()))
        if r_sel != "all":
            sub = sub[sub["R_ratio"] == r_sel]
        if fit_sel != "all":
            sub = sub[sub["fit_type"] == fit_sel]

        curve_fits = fits[fits["curve_id"].isin(sub["curve_id"].unique())]
        params = curve_fits.rename(columns={"log10_C": "log10_C_pred", "m": "m_pred"}).assign(
            model=curve_fits["curve_id"]
        )
        fig = viz.paris_figure(
            f"{alloy}: experimental points + per-curve Paris fits",
            points=sub,
            dk_range=(max(sub["delta_K_ksi_sqrt_in"].min(), 0.5), sub["delta_K_ksi_sqrt_in"].max()),
        )
        for _, row in curve_fits.iterrows():
            dk = np.geomspace(row["dK_min"], row["dK_max"], 40)
            fig.add_scatter(
                x=dk, y=viz.paris_curve(dk, row["log10_C"], row["m"]),
                mode="lines", line=dict(width=1.5, color="#3987e5"), opacity=0.5,
                name=row["curve_id"], showlegend=False,
                hovertemplate=f"{row['curve_id']}: C={row['C']:.2e}, m={row['m']:.2f}<extra></extra>",
            )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Fitted Paris parameters")
        show_cols = ["curve_id", "R_ratio", "C", "m", "r2", "dK_min", "dK_max", "n_points_used", "train_ok"]
        st.dataframe(curve_fits[show_cols].round(4), hide_index=True)
        ref_here = ref[ref["page"].isin(sub["page"].unique())] if len(ref) else ref
        if len(ref_here):
            st.markdown("**Published fits on these pages (book values):**")
            st.dataframe(ref_here.drop(columns=["source_text"]), hide_index=True)

with tab_models:
    metrics_path = Path("reports/metrics.csv")
    loao = _require("loao_predictions")
    preds = _require("predictions")
    if metrics_path.exists() and loao is not None and preds is not None:
        st.subheader("Cross-validation metrics by split scheme")
        metrics = pd.read_csv(metrics_path)
        st.dataframe(
            metrics.pivot_table(index=["scheme", "model"], columns="target", values=["r2", "mae"]).round(3),
        )
        st.caption(
            "Random split is optimistic; group split removes same-experiment leakage; "
            "leave-one-alloy-out is the honest test for predicting a new alloy."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.loao_scatter(loao, "log10_C"), width="stretch")
        with c2:
            st.plotly_chart(viz.loao_scatter(loao, "m"), width="stretch")

        st.subheader("Predicted Paris curves for target alloys")
        intervals = _load("prediction_intervals")
        fits = _load("paris_fits")
        c1, c2 = st.columns(2)
        alloy = c1.selectbox("Target alloy", sorted(preds["alloy"].unique()))
        r_ratio = c2.selectbox("R ratio", sorted(preds[preds.alloy == alloy]["R_ratio"].unique()))
        sel = preds[(preds.alloy == alloy) & (preds.R_ratio == r_ratio)]
        iv = intervals[(intervals.alloy == alloy) & (intervals.R_ratio == r_ratio)]
        st.plotly_chart(
            viz.paris_figure(
                f"Predicted Paris curve — {alloy}, R = {r_ratio} (band = q05–q95 spread)",
                model_params=sel,
                intervals=iv,
                dk_range=applicability_range(fits, alloy),
            ),
            width="stretch",
        )
        tbl = sel.copy()
        tbl["C_pred"] = 10.0 ** tbl["log10_C_pred"]
        st.dataframe(tbl[["model", "C_pred", "m_pred"]].round(4), hide_index=True)

with tab_audit:
    audit = _require("physics_audit")
    if audit is not None:
        pred_audit_path = PROCESSED / "prediction_audit.parquet"
        if pred_audit_path.exists():
            audit = pd.concat([audit, _load("prediction_audit")], ignore_index=True)
        counts = audit.groupby("level").size()
        c1, c2, c3 = st.columns(3)
        c1.metric("PASS", int(counts.get("PASS", 0)))
        c2.metric("WARN", int(counts.get("WARN", 0)))
        c3.metric("FAIL", int(counts.get("FAIL", 0)))

        level = st.selectbox("Show level", ["all", "FAIL", "WARN", "PASS"])
        view = audit if level == "all" else audit[audit["level"] == level]
        st.dataframe(view, hide_index=True, height=450)

        flagged_path = PROCESSED / "flagged_points.parquet"
        if flagged_path.exists():
            flagged = _load("flagged_points")
            if len(flagged):
                st.subheader(f"Excluded source points ({len(flagged)})")
                st.caption("Physically impossible values found in the source workbook (digitization errors).")
                st.dataframe(
                    flagged[["curve_id", "alloy", "delta_K_ksi_sqrt_in", "dadN_in_per_cycle", "flag_reason"]],
                    hide_index=True,
                )

REPORT_FILES = [
    ("predictions.xlsx", "Predicted Paris parameters per alloy/R/model, with spread bands and applicability ranges"),
    ("ai_ready_dataset.xlsx", "Converted dataset in the AI-Ready multi-sheet structure"),
    ("metrics.csv", "Cross-validation metrics per split scheme, model, and target"),
    ("coverage_matrix.csv", "Data coverage by alloy"),
    ("fit_crosscheck.csv", "Our Paris exponents vs the handbook's published fits"),
    ("data_quality.txt", "Conversion log: missing values, duplicates, excluded points"),
]

_MIME = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def _report_zip(reports_dir: Path) -> bytes:
    """Zip every report file, the file-guide READMEs, and the figures."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(reports_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(reports_dir))
    return buf.getvalue()


with tab_downloads:
    reports_dir = Path("reports")
    available = [(n, d) for n, d in REPORT_FILES if (reports_dir / n).exists()]
    if not available:
        st.warning("No report files found — run `uv run python pipeline.py all` first.")
    else:
        st.subheader("Full report")
        n_figures = len(list((reports_dir / "figures").glob("*.png"))) if (reports_dir / "figures").exists() else 0
        st.download_button(
            f"Download everything ({len(available)} files + {n_figures} figures, zip)",
            data=_report_zip(reports_dir),
            file_name="aluminum_fatigue_report.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
            type="primary",
        )
        st.caption(
            "The archive includes README.md / README.uk.md explaining the calculation "
            "and every file's contents."
        )

        st.subheader("Individual files")
        for name, description in available:
            path = reports_dir / name
            col_btn, col_desc = st.columns([1, 2], vertical_alignment="center")
            col_btn.download_button(
                name,
                data=path.read_bytes(),
                file_name=name,
                mime=_MIME.get(path.suffix, "application/octet-stream"),
                icon=":material/download:",
                key=f"dl_{name}",
                width="stretch",
            )
            col_desc.caption(description)
