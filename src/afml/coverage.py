"""Coverage matrix: which alloys have which data, queried with DuckDB over Parquet.

The matrix drives target-alloy selection: an alloy with static properties but no
(or sparse) crack-growth curves is a prediction target; alloys with rich curves
are the training pool and LOAO validation candidates.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from afml.convert import PROCESSED


def coverage_matrix(processed: Path = PROCESSED) -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
    WITH alloys AS (
        SELECT DISTINCT alloy FROM read_parquet('{processed}/materials.parquet')
        UNION
        SELECT DISTINCT alloy FROM read_parquet('{processed}/crack_growth_points.parquet')
    ),
    static AS (
        SELECT m.alloy,
               count(DISTINCT mp.property_name) AS n_static_props,
               count(DISTINCT m.material_id)    AS n_material_variants
        FROM read_parquet('{processed}/materials.parquet') m
        JOIN read_parquet('{processed}/mechanical_properties.parquet') mp USING (material_id)
        GROUP BY m.alloy
    ),
    cg AS (
        SELECT alloy,
               count(*)                   AS n_cg_points,
               count(DISTINCT curve_id)   AS n_cg_curves,
               count(DISTINCT dataset_id) AS n_cg_datasets,
               round(min(R_ratio), 3)     AS cg_R_min,
               round(max(R_ratio), 3)     AS cg_R_max,
               round(min(delta_K_ksi_sqrt_in), 2) AS cg_dK_min,
               round(max(delta_K_ksi_sqrt_in), 2) AS cg_dK_max
        FROM read_parquet('{processed}/crack_growth_points.parquet')
        GROUP BY alloy
    ),
    sn AS (
        SELECT alloy, count(*) AS n_sn_points, count(DISTINCT kt) AS n_sn_kt
        FROM read_parquet('{processed}/sn_points.parquet')
        GROUP BY alloy
    ),
    dt AS (
        SELECT alloy, count(*) AS n_dt_points
        FROM read_parquet('{processed}/dt_data.parquet')
        GROUP BY alloy
    )
    SELECT a.alloy,
           coalesce(s.n_static_props, 0)   AS n_static_props,
           coalesce(s.n_material_variants, 0) AS n_material_variants,
           coalesce(c.n_cg_points, 0)      AS n_cg_points,
           coalesce(c.n_cg_curves, 0)      AS n_cg_curves,
           coalesce(c.n_cg_datasets, 0)    AS n_cg_datasets,
           c.cg_R_min, c.cg_R_max, c.cg_dK_min, c.cg_dK_max,
           coalesce(n.n_sn_points, 0)      AS n_sn_points,
           coalesce(n.n_sn_kt, 0)          AS n_sn_kt,
           coalesce(d.n_dt_points, 0)      AS n_dt_points
    FROM alloys a
    LEFT JOIN static s USING (alloy)
    LEFT JOIN cg c USING (alloy)
    LEFT JOIN sn n USING (alloy)
    LEFT JOIN dt d USING (alloy)
    ORDER BY a.alloy
    """
    return con.execute(query).df()


def suggest_targets(matrix: pd.DataFrame) -> list[str]:
    """Alloys with static properties but no crack-growth curves — candidates for prediction."""
    mask = (matrix["n_static_props"] > 0) & (matrix["n_cg_curves"] == 0)
    return sorted(matrix.loc[mask, "alloy"])
