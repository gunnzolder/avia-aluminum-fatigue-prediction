"""Convert the raw crack-growth workbook into a tidy points table.

Data sheet layout (one per handbook page, with varying blank rows/columns):
    - a Russian/Ukrainian source-note row
    - a row of R-ratio labels ("R - 0.050"), one above each column group
    - a header row (Delta K [ksi*SQRT(in)], da/dN [in/cycle], scaled duplicate)
    - numeric points

The header row is located dynamically (first row containing a "Delta K" cell);
the R-label row is the one directly above it. The third column of each group is
a scaled duplicate of da/dN and is dropped.

R-label sign convention: labels are inconsistent across sheets — "R - 0.500"
means +0.5 on positive-R figures but -0.5 on some negative-R figures, while
other sheets write "R -0.500" with the minus bound to the number. The figure
title's "WITH NEGATIVE R VALUES" marker is therefore authoritative: on those
figures every parsed R is forced negative.
"""

from __future__ import annotations

import re

import pandas as pd
from openpyxl import load_workbook

from afml.convert.chart_meta import FiguresParse, figures_table

# A separator is "=" or a dash followed by whitespace ("R - 0.500" → +0.5);
# a dash bound to the digits is a minus sign ("R -1.000", "R=-0.330" → negative).
# Decimal commas ("R=0,500") are normalized before matching.
_R_LABEL_RE = re.compile(r"R\s*(?:=|-(?=\s))?\s*(-?\d+(?:\.\d+)?)")
_EXPLICIT_MINUS_RE = re.compile(r"-(?=\d)")
_DELTA_K_RE = re.compile(r"delta\s*k", re.IGNORECASE)

META_COLS = ["alloy", "temper", "form", "orientation", "nasgro_code", "figure_id", "fit_type"]


def _normalize(label: object) -> str:
    return str(label).strip().replace(",", ".")


def parse_r_label(label: object) -> float | None:
    if label is None:
        return None
    m = _R_LABEL_RE.search(_normalize(label))
    return float(m.group(1)) if m else None


def has_explicit_minus(label: object) -> bool:
    return label is not None and bool(_EXPLICIT_MINUS_RE.search(_normalize(label)))


def _parse_data_sheet(rows: list[tuple], page: int) -> tuple[pd.DataFrame, bool]:
    """Returns (points, sign_aware). sign_aware = some R label carries an explicit
    minus, meaning the sheet's labels can be trusted as signed."""
    header_idx = next(
        (
            i
            for i, row in enumerate(rows)
            if any(isinstance(c, str) and _DELTA_K_RE.search(c) for c in row)
        ),
        None,
    )
    if header_idx is None or header_idx == 0:
        return pd.DataFrame(), False
    headers = rows[header_idx]
    r_labels = rows[header_idx - 1]

    starts = [i for i, h in enumerate(headers) if isinstance(h, str) and _DELTA_K_RE.search(h)]
    labels = [r_labels[s] if s < len(r_labels) else None for s in starts]
    sign_aware = any(has_explicit_minus(lab) for lab in labels)
    out = []
    for start, label in zip(starts, labels):
        r_ratio = parse_r_label(label)
        for row in rows[header_idx + 1 :]:
            dk = row[start] if start < len(row) else None
            dadn = row[start + 1] if start + 1 < len(row) else None
            if isinstance(dk, (int, float)) and isinstance(dadn, (int, float)):
                out.append({"page": page, "R_ratio": r_ratio, "delta_K": float(dk), "dadN": float(dadn)})
    return pd.DataFrame(out), sign_aware


def crack_growth_points(xlsx_path: str, figures: FiguresParse | None = None) -> pd.DataFrame:
    """Parse every data sheet and attach figure metadata.

    Returns tidy points with keys:
      dataset_id  – physical experiment (nasgro_code + orientation)
      curve_id    – one da/dN–ΔK curve (page + R ratio)
    """
    figures = figures or figures_table(xlsx_path)
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    frames = []
    for fig in figures.figures.itertuples():
        rows = list(wb[fig.data_sheet].iter_rows(values_only=True))
        pts, sign_aware = _parse_data_sheet(rows, fig.page)
        if pts.empty:
            continue
        if getattr(fig, "r_sign", None) == "NEGATIVE" and not sign_aware:
            pts["R_ratio"] = -pts["R_ratio"].abs()
        pts["dataset_id"] = fig.dataset_id
        for col in META_COLS:
            pts[col] = getattr(fig, col)
        frames.append(pts)
    wb.close()
    df = pd.concat(frames, ignore_index=True)
    df["curve_id"] = "p" + df["page"].astype(str) + "_R" + df["R_ratio"].astype(str)
    return df.rename(columns={"delta_K": "delta_K_ksi_sqrt_in", "dadN": "dadN_in_per_cycle"})
