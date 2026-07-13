"""Extract figure/alloy metadata from chart XML inside the raw crack-growth workbook.

The data sheets carry no alloy identity; it lives only in chart titles like

    FIGURE B-1. NASGRO EQUATION FOR FIT 2014-T6 SHEET, T-L (M2AD12AB1)
    WALKER EQUATION FIT FOR 2024-T351 PLATE, L-T (M2EB11AB1) WITH POSITIVE R VALUES

Not every title carries the FIGURE prefix, so charts are associated with their
figure sheet through the OOXML relationship chain (chartsheet -> drawing ->
chart), and figure_id/page come from the sheet name. Figure sheets and data
sheets alternate in the workbook, so each FIGURE sheet is paired with the sheet
that immediately follows it (this also handles the stray data sheet 'Лист7').
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field

import pandas as pd

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

_TITLE_RE = re.compile(
    r"(?:FIGURE\s+B-\d+[a-z]?\.?\s*)?"
    r"(?P<fit_type>NASGRO|WALKER|SPLINE)"
    r".*?(?:FOR\s+FIT|FIT\s+FOR|FOR)\s+"
    r"(?P<alloy>\d{4})-(?P<temper>[A-Z0-9/]+)"
    r"(?:\s+(?P<form>[^,]+?))?\s*,\s*"
    r"(?P<orientation>[A-Z]-[A-Z])\s*"
    r"\((?P<nasgro_code>[A-Z0-9]+)\s*\)"
    r"(?:\s*WITH\s+(?P<r_sign>POSITIVE|NEGATIVE)\s+R\s+VALUES?)?",
    re.IGNORECASE | re.DOTALL,
)

_FIGURE_SHEET_RE = re.compile(
    r"FIGURE\s+(?P<figure_id>B-\d+[a-z]?)\s*;\s*p\.?\s*(?P<page>\d+)", re.IGNORECASE
)


@dataclass
class FiguresParse:
    figures: pd.DataFrame
    unparsed_titles: list[str] = field(default_factory=list)


def _rels(zf: zipfile.ZipFile, part: str) -> dict[str, str]:
    """Relationship id -> target path (resolved relative to the part's directory)."""
    base, fname = part.rsplit("/", 1)
    rels_path = f"{base}/_rels/{fname}.rels"
    if rels_path not in zf.namelist():
        return {}
    out = {}
    for rel in ET.fromstring(zf.read(rels_path)):
        target = rel.get("Target")
        while target.startswith("../"):
            base = base.rsplit("/", 1)[0]
            target = target[3:]
        out[rel.get("Id")] = f"{base}/{target}" if not target.startswith("xl/") else target
    return out


def _sheet_parts(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Workbook-ordered (sheet name, part path like xl/chartsheets/sheet1.xml)."""
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    wb_rels = _rels(zf, "xl/workbook.xml")
    return [
        (sh.get("name"), wb_rels[sh.get(f"{{{_NS_R}}}id")])
        for sh in wb.iter(f"{{{_NS_MAIN}}}sheet")
    ]


def _drawing_parts(zf: zipfile.ZipFile, sheet_part: str) -> list[str]:
    return [t for t in _rels(zf, sheet_part).values() if "/drawings/" in t]


def _chart_parts(zf: zipfile.ZipFile, drawing_part: str) -> list[str]:
    return [t for t in _rels(zf, drawing_part).values() if "/charts/" in t]


def _chart_title(zf: zipfile.ZipFile, chart_part: str) -> str | None:
    xml = zf.read(chart_part).decode("utf-8", "ignore")
    m = re.search(r"<c:title>.*?</c:title>", xml, re.DOTALL)
    if not m:
        return None
    runs = re.findall(r"<a:t>([^<]*)</a:t>", m.group(0))
    title = " ".join(" ".join(runs).split())
    return title or None


def sheet_chart_titles(zf: zipfile.ZipFile) -> dict[str, str]:
    """Figure-sheet name -> chart title, via chartsheet -> drawing -> chart rels."""
    titles: dict[str, str] = {}
    for name, part in _sheet_parts(zf):
        if not _FIGURE_SHEET_RE.search(name):
            continue
        for drawing in _drawing_parts(zf, part):
            for chart in _chart_parts(zf, drawing):
                title = _chart_title(zf, chart)
                if title:
                    titles[name] = title
                    break
            if name in titles:
                break
    return titles


def parse_title(title: str) -> dict | None:
    m = _TITLE_RE.search(title)
    if not m:
        return None
    d = m.groupdict()
    return {
        "fit_type": d["fit_type"].upper(),
        "alloy": d["alloy"],
        "temper": d["temper"].upper(),
        "form": " ".join(d["form"].upper().split()) if d["form"] else None,
        "orientation": d["orientation"].upper(),
        "nasgro_code": d["nasgro_code"].upper(),
        "r_sign": (d["r_sign"] or "").upper() or None,
    }


def figures_table(xlsx_path: str) -> FiguresParse:
    """One row per figure: sheet pairing + chart-title metadata.

    dataset_id identifies the physical experiment (NASGRO code + orientation), so the
    same data digitized on several fit-figures shares one dataset_id.
    """
    with zipfile.ZipFile(xlsx_path) as zf:
        parts = _sheet_parts(zf)
        titles = sheet_chart_titles(zf)

    names = [name for name, _ in parts]
    rows, unparsed = [], []
    for i, name in enumerate(names):
        m = _FIGURE_SHEET_RE.search(name)
        if not m:
            continue
        if i + 1 >= len(names) or _FIGURE_SHEET_RE.search(names[i + 1]):
            raise ValueError(f"Figure sheet {name!r} has no following data sheet")
        row = {
            "figure_id": m["figure_id"],
            "page": int(m["page"]),
            "figure_sheet": name,
            "data_sheet": names[i + 1],
            "title": titles.get(name),
        }
        meta = parse_title(row["title"]) if row["title"] else None
        if meta is None and row["title"]:
            unparsed.append(row["title"])
        row.update(meta or {})
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("fit_type", "alloy", "temper", "form", "orientation", "nasgro_code", "r_sign"):
        if col not in df.columns:
            df[col] = None
    df["dataset_id"] = df["nasgro_code"].fillna("UNKNOWN") + "_" + df["orientation"].fillna("NA")
    return FiguresParse(figures=df, unparsed_titles=unparsed)
