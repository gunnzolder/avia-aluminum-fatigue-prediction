"""Plotly figures shared by the Streamlit dashboard and the PNG report export.

Colors follow the entity, never the rank: each model keeps its slot everywhere.
Palette is the validated default categorical set (see dataviz skill palette.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODEL_COLORS = {  # fixed categorical slots
    "ridge": "#2a78d6",
    "random_forest": "#1baf7a",
    "svr": "#eda100",
}
MODEL_LABELS = {"ridge": "Ridge regression", "random_forest": "Random Forest", "svr": "SVR (RBF)"}
POINT_COLOR = "#6b6a63"  # muted ink for experimental points
REF_COLOR = "#4a3aa7"  # violet for published reference fits

_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=13),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=70, r=30, t=60, b=60),
)


def paris_curve(dk: np.ndarray, log10_c: float, m: float) -> np.ndarray:
    return 10.0**log10_c * dk**m


def paris_figure(
    title: str,
    points: pd.DataFrame | None = None,
    model_params: pd.DataFrame | None = None,
    intervals: pd.DataFrame | None = None,
    dk_range: tuple[float, float] = (2.0, 40.0),
) -> go.Figure:
    """Log-log da/dN–ΔK chart: experimental points + per-model curves + spread band.

    model_params columns: model, log10_C_pred, m_pred.
    intervals (optional) tidy: model, target, q05, q95 — used for the band.
    """
    fig = go.Figure()
    dk = np.geomspace(dk_range[0], dk_range[1], 60)

    if points is not None and len(points):
        fig.add_trace(
            go.Scatter(
                x=points["delta_K_ksi_sqrt_in"],
                y=points["dadN_in_per_cycle"],
                mode="markers",
                marker=dict(color=POINT_COLOR, size=6, opacity=0.55),
                name="Experimental points",
            )
        )

    for _, row in (model_params if model_params is not None else pd.DataFrame()).iterrows():
        model = row["model"]
        color = MODEL_COLORS.get(model, POINT_COLOR)
        if intervals is not None and len(intervals):
            iv = intervals[intervals["model"] == model]
            if len(iv) == 2:
                c_iv = iv[iv["target"] == "log10_C"].iloc[0]
                m_iv = iv[iv["target"] == "m"].iloc[0]
                lo = np.minimum(
                    paris_curve(dk, c_iv["q05"], m_iv["q05"]), paris_curve(dk, c_iv["q05"], m_iv["q95"])
                )
                hi = np.maximum(
                    paris_curve(dk, c_iv["q95"], m_iv["q95"]), paris_curve(dk, c_iv["q95"], m_iv["q05"])
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([dk, dk[::-1]]),
                        y=np.concatenate([hi, lo[::-1]]),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.12,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=dk,
                y=paris_curve(dk, row["log10_C_pred"], row["m_pred"]),
                mode="lines",
                line=dict(color=color, width=2),
                name=MODEL_LABELS.get(model, model),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(type="log", title="ΔK [ksi·√in]"),
        yaxis=dict(type="log", title="da/dN [in/cycle]"),
        **_LAYOUT,
    )
    return fig


def loao_scatter(loao: pd.DataFrame, target: str) -> go.Figure:
    """Predicted vs actual per model for one target, colored by model (LOAO folds)."""
    fig = go.Figure()
    t, p = f"{target}_true", f"{target}_pred"
    lo = min(loao[t].min(), loao[p].min())
    hi = max(loao[t].max(), loao[p].max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#c3c2b7", dash="dash", width=1), showlegend=False, hoverinfo="skip",
        )
    )
    for model, grp in loao.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=grp[t], y=grp[p], mode="markers",
                marker=dict(color=MODEL_COLORS.get(model), size=7, opacity=0.7),
                name=MODEL_LABELS.get(model, model),
                customdata=grp[["alloy", "curve_id"]],
                hovertemplate="%{customdata[0]} %{customdata[1]}<br>true %{x:.2f} → pred %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Leave-one-alloy-out: {target}",
        xaxis_title=f"{target} (fit from data)",
        yaxis_title=f"{target} (predicted)",
        **_LAYOUT,
    )
    return fig


def coverage_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Alloy × data-kind presence heatmap (sequential blue, light→dark)."""
    cols = {
        "n_static_props": "Static props",
        "n_cg_curves": "da/dN curves",
        "n_sn_points": "S-N points",
        "n_dt_points": "DT data",
    }
    z = matrix[list(cols)].to_numpy(dtype=float)
    z_norm = np.log10(z + 1)
    fig = go.Figure(
        go.Heatmap(
            z=z_norm,
            x=list(cols.values()),
            y=matrix["alloy"],
            colorscale=[[0, "#f4f4ef"], [0.25, "#cde2fb"], [0.55, "#86b6ef"], [0.8, "#3987e5"], [1, "#1c5cab"]],
            text=z.astype(int).astype(str),
            texttemplate="%{text}",
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(title="Data coverage by alloy (counts)", **_LAYOUT)
    fig.update_yaxes(autorange="reversed")
    return fig
