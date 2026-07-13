"""Per-curve Paris law fits: da/dN = C * (ΔK)^m, fitted as log-log OLS.

The Paris (power-law) region is the middle segment of a da/dN–ΔK curve; the
near-threshold and fast-fracture tails bend away from the straight line in
log-log space. Tails are trimmed iteratively: fit all points, drop the worst
end-point residual, refit — while r² improves meaningfully and enough points
remain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_POINTS = 5
R2_IMPROVEMENT_EPS = 0.002


def _ols_loglog(log_dk: np.ndarray, log_dadn: np.ndarray) -> tuple[float, float, float]:
    """Returns (log10_C, m, r2)."""
    m, intercept = np.polyfit(log_dk, log_dadn, 1)
    pred = m * log_dk + intercept
    ss_res = float(np.sum((log_dadn - pred) ** 2))
    ss_tot = float(np.sum((log_dadn - log_dadn.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(intercept), float(m), r2


def fit_paris_curve(delta_k: np.ndarray, dadn: np.ndarray) -> dict | None:
    """Fit one curve, trimming tail points that bend away from the Paris line."""
    order = np.argsort(delta_k)
    log_dk, log_dadn = np.log10(delta_k[order]), np.log10(dadn[order])
    if len(log_dk) < MIN_POINTS:
        return None

    lo, hi = 0, len(log_dk)
    log_c, m, r2 = _ols_loglog(log_dk, log_dadn)
    while hi - lo > MIN_POINTS:
        candidates = []
        for nlo, nhi in ((lo + 1, hi), (lo, hi - 1)):
            c2, m2, r2_2 = _ols_loglog(log_dk[nlo:nhi], log_dadn[nlo:nhi])
            candidates.append((r2_2, nlo, nhi, c2, m2))
        best = max(candidates)
        if best[0] - r2 < R2_IMPROVEMENT_EPS:
            break
        r2, lo, hi, log_c, m = best[0], best[1], best[2], best[3], best[4]

    return {
        "log10_C": log_c,
        "C": 10.0**log_c,
        "m": m,
        "r2": r2,
        "n_points_used": hi - lo,
        "n_points_total": len(log_dk),
        "dK_min": float(10 ** log_dk[lo]),
        "dK_max": float(10 ** log_dk[hi - 1]),
    }


def fit_all_curves(points: pd.DataFrame) -> pd.DataFrame:
    """Fit every curve_id; returns one row per successfully fitted curve."""
    meta_cols = ["dataset_id", "page", "alloy", "temper", "form", "orientation", "R_ratio", "fit_type"]
    rows = []
    for curve_id, grp in points.groupby("curve_id"):
        fit = fit_paris_curve(
            grp["delta_K_ksi_sqrt_in"].to_numpy(), grp["dadN_in_per_cycle"].to_numpy()
        )
        if fit is None:
            continue
        row = {"curve_id": curve_id, **grp.iloc[0][meta_cols].to_dict(), **fit}
        rows.append(row)
    return pd.DataFrame(rows)


def crosscheck_reference(fits: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Compare our per-curve m against the book's published exponent n on the same page.

    The book's C belongs to NASGRO/Walker forms (R-normalized), so only the
    exponent is directly comparable, and only loosely — this is a sanity check,
    not a metric.
    """
    if reference.empty:
        return pd.DataFrame()
    merged = fits.merge(reference[["page", "ref_C", "ref_n"]], on="page", how="inner")
    merged["m_vs_ref_n_pct"] = 100.0 * (merged["m"] - merged["ref_n"]) / merged["ref_n"]
    return merged[["curve_id", "page", "alloy", "R_ratio", "m", "ref_n", "m_vs_ref_n_pct", "C", "ref_C", "r2"]]
