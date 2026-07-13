"""Converters: raw xlsx sources → validated tidy Parquet tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from afml import schemas
from afml.convert import ai_ready
from afml.convert.chart_meta import figures_table
from afml.convert.crack_growth import crack_growth_points
from afml.convert.ref_fits import reference_fits

RAW_DB = Path("inbox/Fatigue Crack Growth Database.xlsx")
AI_READY = Path("inbox/Aluminum_Data_AI_Ready_mod.xlsx")
PROCESSED = Path("data/processed")


def convert_all(
    raw_db: Path = RAW_DB, ai_ready_path: Path = AI_READY, out_dir: Path = PROCESSED
) -> list[schemas.QualityReport]:
    """Run every converter, validate, write Parquet, return quality reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[schemas.QualityReport] = []

    fp = figures_table(str(raw_db))
    fp.figures.to_parquet(out_dir / "figures.parquet")

    points = crack_growth_points(str(raw_db), fp)
    points, n_dupes = schemas.dedupe_points(points)
    points, flagged = schemas.physical_filter(points)
    schemas.crack_growth_schema.validate(points, lazy=True)
    points.to_parquet(out_dir / "crack_growth_points.parquet")
    flagged.to_parquet(out_dir / "flagged_points.parquet")
    report = schemas.quality_report("crack_growth_points", points, n_dupes)
    if len(flagged):
        report.notes.append(
            f"{len(flagged)} physically impossible points flagged and excluded "
            f"(pages {sorted(flagged['page'].unique())})"
        )
    reports.append(report)

    ref = reference_fits(str(raw_db))
    ref.to_parquet(out_dir / "reference_fits.parquet")
    reports.append(schemas.quality_report("reference_fits", ref))

    mat = ai_ready.materials(str(ai_ready_path))
    mat.to_parquet(out_dir / "materials.parquet")
    reports.append(schemas.quality_report("materials", mat))

    mech = ai_ready.mechanical_properties(str(ai_ready_path))
    schemas.mech_props_schema.validate(mech, lazy=True)
    mech.to_parquet(out_dir / "mechanical_properties.parquet")
    reports.append(schemas.quality_report("mechanical_properties", mech))

    phys = ai_ready.physical_properties(str(ai_ready_path))
    phys.to_parquet(out_dir / "physical_properties.parquet")
    reports.append(schemas.quality_report("physical_properties", phys))

    sn = ai_ready.sn_points(str(ai_ready_path))
    schemas.sn_schema.validate(sn, lazy=True)
    sn.to_parquet(out_dir / "sn_points.parquet")
    reports.append(schemas.quality_report("sn_points", sn))

    dt = ai_ready.dt_data(str(ai_ready_path))
    dt.to_parquet(out_dir / "dt_data.parquet")
    reports.append(schemas.quality_report("dt_data", dt))

    if fp.unparsed_titles:
        reports[0].notes.append(f"unparsed chart titles: {fp.unparsed_titles}")
    return reports


def load(table: str, out_dir: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(out_dir / f"{table}.parquet")
