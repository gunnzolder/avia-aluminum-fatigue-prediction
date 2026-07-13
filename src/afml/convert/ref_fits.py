"""Harvest published NASGRO/Walker fit coefficients from drawing text boxes.

Figure sheets contain text boxes like

    NASGRO 4.0 WALKER FIT:  C = 0.246E-08  n = 3.376  m+ = 0.750

These are the book's own fits — used as reference values to cross-check our
Paris fits, not as training data.
"""

from __future__ import annotations

import re
import zipfile

import pandas as pd

from afml.convert.chart_meta import (
    _FIGURE_SHEET_RE,
    _chart_parts,
    _drawing_parts,
    _rels,
    _sheet_parts,
)

_C_RE = re.compile(r"\bC\s*=\s*([0-9.]+\s*E\s*[-+]?\d+)", re.IGNORECASE)
_N_RE = re.compile(r"\bn\s*=\s*([0-9.]+)", re.IGNORECASE)
_M_RE = re.compile(r"\bm\s*([+-]?)\s*=\s*(-?[0-9.]+)")


def _drawing_text(zf: zipfile.ZipFile, sheet_part: str) -> str:
    """Text of chart userShapes: chartsheet -> drawing -> chart -> userShapes drawing."""
    chunks = []
    for drawing in _drawing_parts(zf, sheet_part):
        for chart in _chart_parts(zf, drawing):
            for target in _rels(zf, chart).values():
                if "drawing" not in target or target not in zf.namelist():
                    continue
                xml = zf.read(target).decode("utf-8", "ignore")
                chunks += re.findall(r"<a:t>([^<]*)</a:t>", xml)
    return " ".join(" ".join(chunks).split())


def reference_fits(xlsx_path: str) -> pd.DataFrame:
    """One row per figure sheet whose drawings contain fit coefficients."""
    rows = []
    with zipfile.ZipFile(xlsx_path) as zf:
        for name, part in _sheet_parts(zf):
            m_sheet = _FIGURE_SHEET_RE.search(name)
            if not m_sheet:
                continue
            text = _drawing_text(zf, part)
            if not text:
                continue
            c, n = _C_RE.search(text), _N_RE.search(text)
            if not (c and n):
                continue
            walker = {
                f"walker_m_{'minus' if s == '-' else 'plus'}": float(v)
                for s, v in _M_RE.findall(text)
            }
            rows.append(
                {
                    "figure_id": m_sheet["figure_id"],
                    "page": int(m_sheet["page"]),
                    "ref_C": float(c.group(1).replace(" ", "")),
                    "ref_n": float(n.group(1)),
                    "walker_m_plus": walker.get("walker_m_plus"),
                    "walker_m_minus": walker.get("walker_m_minus"),
                    "source_text": text[:200],
                }
            )
    return pd.DataFrame(rows)
