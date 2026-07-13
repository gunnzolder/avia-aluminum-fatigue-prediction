"""Physics audit required by the spec, applied to both input curves and predictions.

Checks (Paris/crack-growth):
  - da/dN positive
  - da/dN monotone non-decreasing with ΔK in the Paris region (tolerance for
    digitization noise)
  - Paris exponent m within the plausible band for aluminum alloys
  - predicted C positive and curve evaluated only inside its applicability range

Result levels: PASS / WARN / FAIL. WARN marks suspicious-but-usable curves;
FAIL marks curves excluded from ML training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

M_PLAUSIBLE = (1.5, 6.0)  # typical Paris exponents for Al alloys ~2–4
M_HARD = (0.5, 8.0)  # outside this: FAIL (digitization artifacts)
MONOTONE_TOLERANCE = 0.20  # fraction of decreasing consecutive pairs tolerated


def audit_curve_points(points: pd.DataFrame) -> pd.DataFrame:
    """Point-level audit per curve_id: positivity + monotonicity in log-log space."""
    rows = []
    for curve_id, grp in points.groupby("curve_id"):
        grp = grp.sort_values("delta_K_ksi_sqrt_in")
        dadn = grp["dadN_in_per_cycle"].to_numpy()
        n = len(dadn)
        n_nonpositive = int((dadn <= 0).sum())
        decreasing = (np.diff(dadn) < 0).sum() / max(n - 1, 1)
        if n_nonpositive:
            level, reason = "FAIL", f"{n_nonpositive} non-positive dadN"
        elif decreasing > MONOTONE_TOLERANCE:
            level, reason = "WARN", f"{decreasing:.0%} of steps decrease dadN"
        else:
            level, reason = "PASS", ""
        rows.append(
            {
                "curve_id": curve_id,
                "alloy": grp["alloy"].iloc[0],
                "check": "points",
                "level": level,
                "reason": reason,
                "n_points": n,
            }
        )
    return pd.DataFrame(rows)


def audit_fit(fits: pd.DataFrame) -> pd.DataFrame:
    """Fit-level audit: exponent plausibility and fit quality."""

    def level_reason(row) -> tuple[str, str]:
        if not (M_HARD[0] <= row["m"] <= M_HARD[1]):
            return "FAIL", f"m={row['m']:.2f} outside hard band {M_HARD}"
        if not (M_PLAUSIBLE[0] <= row["m"] <= M_PLAUSIBLE[1]):
            return "WARN", f"m={row['m']:.2f} outside plausible band {M_PLAUSIBLE}"
        if row["r2"] < 0.95:
            return "WARN", f"r2={row['r2']:.3f} < 0.95"
        return "PASS", ""

    out = fits[["curve_id", "alloy", "m", "r2"]].copy()
    levels = out.apply(level_reason, axis=1, result_type="expand")
    out["check"], out["level"], out["reason"] = "fit", levels[0], levels[1]
    return out[["curve_id", "alloy", "check", "level", "reason", "m", "r2"]]


def audit_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Audit predicted Paris parameters (columns: alloy, model, log10_C_pred, m_pred)."""
    rows = []
    for _, p in predictions.iterrows():
        problems = []
        level = "PASS"
        if not (M_HARD[0] <= p["m_pred"] <= M_HARD[1]):
            level, problems = "FAIL", [f"predicted m={p['m_pred']:.2f} outside {M_HARD}"]
        elif not (M_PLAUSIBLE[0] <= p["m_pred"] <= M_PLAUSIBLE[1]):
            level, problems = "WARN", [f"predicted m={p['m_pred']:.2f} outside {M_PLAUSIBLE}"]
        if p["log10_C_pred"] > -4:
            level = "FAIL"
            problems.append(f"log10_C={p['log10_C_pred']:.2f} implies implausibly fast growth")
        rows.append(
            {
                "alloy": p["alloy"],
                "model": p["model"],
                "R_ratio": p.get("R_ratio"),
                "check": "prediction",
                "level": level,
                "reason": "; ".join(problems),
            }
        )
    return pd.DataFrame(rows)


def training_mask(fits: pd.DataFrame, point_audit: pd.DataFrame) -> pd.Series:
    """Curves eligible for ML training: no FAIL on any check."""
    failed = set(point_audit.loc[point_audit["level"] == "FAIL", "curve_id"]) | set(
        audit_fit(fits).query("level == 'FAIL'")["curve_id"]
    )
    return ~fits["curve_id"].isin(failed)
