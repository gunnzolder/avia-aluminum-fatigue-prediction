"""Feature assembly: one row per fitted curve, joining alloy-level properties.

Targets: (log10_C, m). Features: alloy static strength properties (median over
material variants, basis A/B mixed), physical properties, R ratio, and
alloy-series/orientation categoricals.
"""

from __future__ import annotations

import pandas as pd

# L-direction strength properties used as features.
STRENGTH_PROPS = ["Ftu", "Fty", "Fcy", "Fsu", "e"]
PHYS_PROPS = ["density_lb_in3", "E_primary_10_3_ksi", "G_10_3_ksi", "poissons_ratio"]
TARGETS = ["log10_C", "m"]


def alloy_property_table(
    materials: pd.DataFrame, mech: pd.DataFrame, phys: pd.DataFrame
) -> pd.DataFrame:
    """Alloy-level medians of strength and physical properties.

    Median over all material variants (forms/thicknesses/bases) of the alloy: the
    crack-growth curves don't map 1:1 to AI-Ready material_ids, so alloy level is
    the honest join key.
    """
    mech_alloy = (
        mech.merge(materials[["material_id", "alloy"]].drop_duplicates(), on="material_id")
        .query("direction == 'L' and property_name in @STRENGTH_PROPS")
        .groupby(["alloy", "property_name"])["value_ksi"]
        .median()
        .unstack()
        .reset_index()
    )
    phys_alloy = (
        phys.merge(materials[["material_id", "alloy"]].drop_duplicates(), on="material_id")
        .groupby("alloy")[[c for c in PHYS_PROPS if c in phys.columns]]
        .median()
        .reset_index()
    )
    return mech_alloy.merge(phys_alloy, on="alloy", how="outer")


def build_features(fits: pd.DataFrame, alloy_props: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for fitted curves: targets + numeric/categorical features."""
    df = fits.merge(alloy_props, on="alloy", how="left")
    df["alloy_series"] = df["alloy"].str[0] + "xxx"
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric_features, categorical_features) present in the frame."""
    numeric = [c for c in STRENGTH_PROPS + PHYS_PROPS + ["R_ratio"] if c in df.columns]
    categorical = [c for c in ["alloy_series", "orientation"] if c in df.columns]
    return numeric, categorical


def target_feature_rows(alloy: str, alloy_props: pd.DataFrame, r_values: list[float]) -> pd.DataFrame:
    """Feature rows for a prediction-target alloy (no fitted curves) at given R ratios."""
    props = alloy_props[alloy_props["alloy"] == alloy]
    if props.empty:
        raise ValueError(f"no properties for target alloy {alloy}")
    rows = props.loc[props.index.repeat(len(r_values))].reset_index(drop=True)
    rows["R_ratio"] = list(r_values)
    rows["alloy_series"] = alloy[0] + "xxx"
    rows["orientation"] = "L-T"
    return rows
