"""Pandera schemas and data-quality reporting for all processed tables.

Schemas enforce physical plausibility at the pipeline boundary: positive ΔK and
da/dN, R in a sane range, non-null keys, no duplicate points within a curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

crack_growth_schema = pa.DataFrameSchema(
    {
        "curve_id": pa.Column(str, nullable=False),
        "dataset_id": pa.Column(str, nullable=False),
        "page": pa.Column(int, pa.Check.gt(0)),
        "alloy": pa.Column(str, nullable=False),
        "temper": pa.Column(str, nullable=False),
        "orientation": pa.Column(str, nullable=False),
        "R_ratio": pa.Column(float, pa.Check.in_range(-5.0, 1.0, include_max=False)),
        "delta_K_ksi_sqrt_in": pa.Column(float, pa.Check.gt(0)),
        "dadN_in_per_cycle": pa.Column(float, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
)

mech_props_schema = pa.DataFrameSchema(
    {
        "material_id": pa.Column(str, nullable=False),
        "basis": pa.Column(str, nullable=False),
        "property_name": pa.Column(str, nullable=False),
        "direction": pa.Column(str, nullable=True),
        "value_ksi": pa.Column(float, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
)

sn_schema = pa.DataFrameSchema(
    {
        "alloy": pa.Column(str, nullable=False),
        "kt": pa.Column(float, pa.Check.ge(0.5), nullable=True),
        "R_ratio": pa.Column(float, pa.Check.in_range(-5.0, 1.0, include_max=False), nullable=True),
        "max_stress_ksi": pa.Column(float, pa.Check.gt(0)),
        "cycles_to_failure": pa.Column(float, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
)


@dataclass
class QualityReport:
    table: str
    n_rows: int
    n_dropped_duplicates: int
    n_missing_by_col: dict
    notes: list


def dedupe_points(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate points within a curve (repeated digitization)."""
    before = len(df)
    df = df.drop_duplicates(subset=["curve_id", "delta_K_ksi_sqrt_in", "dadN_in_per_cycle"])
    return df.reset_index(drop=True), before - len(df)


# Physical plausibility bounds for aluminum alloys (imperial units).
# Kmax accounts for R: delta_K = (1 - R) * Kmax, so negative-R curves legitimately
# reach large delta_K. Fracture toughness of Al alloys is < ~150 ksi*sqrt(in).
KMAX_BOUND_KSI_SQRT_IN = 200.0
DADN_BOUND_IN_PER_CYCLE = 0.1


def physical_filter(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split crack-growth points into (physically plausible, flagged-impossible).

    Flagged points get a `flag_reason` column; they come from digitization errors
    in the source workbook (e.g. p.90 tail where delta_K runs to 3.5e5).
    """
    kmax = points["delta_K_ksi_sqrt_in"] / (1.0 - points["R_ratio"])
    reasons = pd.Series("", index=points.index, dtype="string")
    reasons[kmax > KMAX_BOUND_KSI_SQRT_IN] += "kmax_exceeds_bound;"
    reasons[points["dadN_in_per_cycle"] > DADN_BOUND_IN_PER_CYCLE] += "dadn_exceeds_bound;"
    bad = reasons != ""
    flagged = points[bad].assign(flag_reason=reasons[bad].str.rstrip(";"))
    return points[~bad].reset_index(drop=True), flagged.reset_index(drop=True)


def quality_report(name: str, df: pd.DataFrame, dropped_dupes: int = 0) -> QualityReport:
    notes = []
    if name == "crack_growth_points":
        per_curve = df.groupby("curve_id").size()
        thin = per_curve[per_curve < 5]
        if len(thin):
            notes.append(f"{len(thin)} curves have <5 points: {list(thin.index[:8])}")
        # same (curve_id, delta_K) with different dadN — conflicting digitization
        conflicts = df.duplicated(subset=["curve_id", "delta_K_ksi_sqrt_in"], keep=False)
        if conflicts.any():
            notes.append(
                f"{int(conflicts.sum())} points share (curve_id, delta_K) with "
                f"conflicting dadN across {df.loc[conflicts, 'curve_id'].nunique()} curves"
            )
    return QualityReport(
        table=name,
        n_rows=len(df),
        n_dropped_duplicates=dropped_dupes,
        n_missing_by_col={c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        notes=notes,
    )
