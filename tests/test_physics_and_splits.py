"""Physics audit rules and split-leakage guarantees."""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from afml import physics
from afml.schemas import physical_filter


def _points(curve_id, alloy, dk, dadn):
    return pd.DataFrame(
        {
            "curve_id": curve_id,
            "alloy": alloy,
            "delta_K_ksi_sqrt_in": dk,
            "dadN_in_per_cycle": dadn,
            "R_ratio": 0.1,
        }
    )


def test_monotone_curve_passes():
    dk = np.geomspace(3, 30, 10)
    audit = physics.audit_curve_points(_points("c1", "2024", dk, 1e-9 * dk**3))
    assert audit.iloc[0]["level"] == "PASS"


def test_heavily_nonmonotone_curve_warns():
    dk = np.geomspace(3, 30, 10)
    dadn = 1e-9 * dk**3
    dadn[::2] *= 0.01  # every other point drops far below the trend
    audit = physics.audit_curve_points(_points("c1", "2024", dk, dadn))
    assert audit.iloc[0]["level"] == "WARN"


def test_fit_audit_bands():
    fits = pd.DataFrame(
        {
            "curve_id": ["ok", "warn", "fail"],
            "alloy": ["2024"] * 3,
            "m": [3.0, 7.0, 17.0],
            "r2": [0.99, 0.99, 0.99],
        }
    )
    audit = physics.audit_fit(fits)
    assert audit.set_index("curve_id")["level"].to_dict() == {
        "ok": "PASS", "warn": "WARN", "fail": "FAIL",
    }


def test_physical_filter_flags_impossible_kmax_respecting_R():
    # Same delta_K: impossible at R=0.1, plausible at R=-3.33
    df = pd.DataFrame(
        {
            "curve_id": ["a", "b"],
            "page": [1, 2],
            "delta_K_ksi_sqrt_in": [209.0, 209.0],
            "dadN_in_per_cycle": [1e-3, 1e-3],
            "R_ratio": [0.1, -3.33],
        }
    )
    clean, flagged = physical_filter(df)
    assert flagged["curve_id"].tolist() == ["a"]
    assert clean["curve_id"].tolist() == ["b"]


def test_loao_never_leaks_target_alloy():
    df = pd.DataFrame({"alloy": ["2024"] * 5 + ["7075"] * 5 + ["7475"] * 5, "x": range(15)})
    for train_idx, test_idx in LeaveOneGroupOut().split(df, groups=df["alloy"]):
        train_alloys = set(df.iloc[train_idx]["alloy"])
        test_alloys = set(df.iloc[test_idx]["alloy"])
        assert train_alloys.isdisjoint(test_alloys)
