"""Data-sheet parsing: layout variants and R-label conventions."""

import pandas as pd

from afml.convert.crack_growth import _parse_data_sheet, parse_r_label


def test_r_label_spaced_dash_is_positive():
    assert parse_r_label("R - 0.500") == 0.5
    assert parse_r_label("R - 0.050") == 0.05


def test_r_label_bound_dash_is_negative():
    assert parse_r_label("R -1.000") == -1.0
    assert parse_r_label("R -  -1.000") == -1.0


def test_r_label_equals_variants():
    assert parse_r_label("R=0.800") == 0.8
    assert parse_r_label("R=-0.330") == -0.33
    assert parse_r_label("R=0,500") == 0.5  # decimal comma
    assert parse_r_label("R=-0,500") == -0.5


def test_explicit_minus_detection():
    from afml.convert.crack_growth import has_explicit_minus

    assert has_explicit_minus("R=-0,500")
    assert has_explicit_minus("R -1.000")
    assert not has_explicit_minus("R - 0.500")
    assert not has_explicit_minus("R=0.800")


def test_r_label_garbage_is_none():
    assert parse_r_label(None) is None
    assert parse_r_label("промежуточные значения") is None


def _sheet(rows):
    return [tuple(r) for r in rows]


def test_parse_sheet_with_leading_blank_rows_and_column_offset():
    rows = _sheet([
        (None, "Данные взяты из книги", None, None, None),
        (None, None, None, None, None),
        (None, "R - 0.100", None, None, None),
        (None, "Delta K [ksi*SQRT(in)]", "d a/d N [in/c y cle]", "scaled", None),
        (None, 3.0, 1e-7, 1.0, None),
        (None, 4.0, 2e-7, 2.0, None),
    ])
    df, sign_aware = _parse_data_sheet(rows, page=99)
    assert len(df) == 2
    assert not sign_aware
    assert df["R_ratio"].unique().tolist() == [0.1]
    assert df["delta_K"].tolist() == [3.0, 4.0]


def test_parse_sheet_multiple_column_groups():
    rows = _sheet([
        ("note", None, None, None, None, None, None),
        ("R - 0.100", None, None, None, "R - 0.500", None, None),
        ("Delta K [x]", "d a/d N", "s", None, "Delta K [x]", "d a/d N", "s"),
        (3.0, 1e-7, 1.0, None, 2.0, 5e-8, 0.5),
    ])
    df, _ = _parse_data_sheet(rows, page=1)
    assert set(df["R_ratio"]) == {0.1, 0.5}
    assert len(df) == 2


def test_parse_sheet_sign_aware_when_explicit_minus_present():
    rows = _sheet([
        ("note", None, None, None, None, None, None),
        ("R=-0,500", None, None, None, "R=0,500", None, None),
        ("Delta K [x]", "d a/d N", "s", None, "Delta K [x]", "d a/d N", "s"),
        (3.0, 1e-7, 1.0, None, 2.0, 5e-8, 0.5),
    ])
    df, sign_aware = _parse_data_sheet(rows, page=55)
    assert sign_aware
    assert set(df["R_ratio"]) == {-0.5, 0.5}


def test_parse_sheet_ignores_text_cells_in_data():
    rows = _sheet([
        ("note",),
        ("R - 0.100", None),
        ("Delta K [x]", "d a/d N"),
        ("not a number", 1e-7),
        (3.0, 1e-7),
    ])
    df, _ = _parse_data_sheet(rows, page=1)
    assert len(df) == 1
