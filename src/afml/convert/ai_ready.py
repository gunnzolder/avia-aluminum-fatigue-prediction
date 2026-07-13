"""Read the reference AI-Ready workbook into tidy tables.

Sheets: materials, mechanical_properties (wide repeated triples → long),
physical_properties, fatigue_data (S-N points), DT_Data.
"""

from __future__ import annotations

import pandas as pd


def _read(xlsx_path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return df.dropna(axis=1, how="all").replace("NaN", pd.NA)


def materials(xlsx_path: str) -> pd.DataFrame:
    df = _read(xlsx_path, "materials")
    df["alloy"] = df["alloy"].astype(str)
    for col in df.columns:
        if col.startswith(("thickness_", "cross_sectional_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif df[col].dtype == object:
            df[col] = df[col].astype("string")
    return df


def mechanical_properties(xlsx_path: str) -> pd.DataFrame:
    """Unpivot repeated (property_name, direction, value_ksi) triples to long format."""
    df = _read(xlsx_path, "mechanical_properties")
    id_cols = ["material_id", "basis"]
    triple_starts = [i for i, c in enumerate(df.columns) if str(c).startswith("property_name")]
    frames = []
    for start in triple_starts:
        cols = df.columns[start : start + 3]
        part = df[id_cols + list(cols)].copy()
        part.columns = id_cols + ["property_name", "direction", "value_ksi"]
        frames.append(part)
    long = pd.concat(frames, ignore_index=True).dropna(subset=["property_name", "value_ksi"])
    long["value_ksi"] = pd.to_numeric(long["value_ksi"], errors="coerce")
    for col in ("material_id", "basis", "property_name", "direction"):
        long[col] = long[col].astype("string")
    return long.dropna(subset=["value_ksi"]).reset_index(drop=True)


def physical_properties(xlsx_path: str) -> pd.DataFrame:
    df = _read(xlsx_path, "physical_properties")
    for col in df.columns:
        if col not in ("material_id", "basis"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sn_points(xlsx_path: str) -> pd.DataFrame:
    df = _read(xlsx_path, "fatigue_data")
    df["alloy"] = df["alloy"].astype(str)
    for col in ("kt", "R_ratio", "mean_stress", "max_stress_ksi", "cycles_to_failure"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["max_stress_ksi", "cycles_to_failure"]).reset_index(drop=True)


def dt_data(xlsx_path: str) -> pd.DataFrame:
    df = _read(xlsx_path, "DT_Data")
    df["alloy"] = df["alloy"].astype(str)
    for col in ("t_in", "Ka"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Ka"]).reset_index(drop=True)
