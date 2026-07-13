"""Chart-title parsing: every real title variant must parse."""

import pytest

from afml.convert.chart_meta import parse_title

REAL_TITLE_VARIANTS = [
    (
        "FIGURE B-1. NASGRO EQUATION FOR FIT 2014-T6 SHEET, T-L (M2AD12AB1)",
        {"fit_type": "NASGRO", "alloy": "2014", "temper": "T6", "form": "SHEET",
         "orientation": "T-L", "nasgro_code": "M2AD12AB1", "r_sign": None},
    ),
    (
        "FIGURE B-6a. SPLINE FIT FOR 2024-T3 CLAD AND BARE SHEET, L-T (M2EA11AB1) WITH POSITIVE R VALUES",
        {"fit_type": "SPLINE", "alloy": "2024", "temper": "T3", "form": "CLAD AND BARE SHEET",
         "orientation": "L-T", "nasgro_code": "M2EA11AB1", "r_sign": "POSITIVE"},
    ),
    (
        "WALKER EQUATION FIT FOR 2024-T351 PLATE, L-T (M2EB11AB1) WITH NEGATIVE R VALUES",
        {"fit_type": "WALKER", "alloy": "2024", "temper": "T351", "form": "PLATE",
         "orientation": "L-T", "nasgro_code": "M2EB11AB1", "r_sign": "NEGATIVE"},
    ),
    # no form between temper and comma
    (
        "FIGURE B-22. NASGRO EQUATION FIT FOR 7050-T7451, L-T (M7GJ11AC1)",
        {"fit_type": "NASGRO", "alloy": "7050", "temper": "T7451", "form": None,
         "orientation": "L-T", "nasgro_code": "M7GJ11AC1", "r_sign": None},
    ),
    # trailing space inside parens; singular "R VALUE"
    (
        "SPLINE FIT FOR 2024-T351 PLATE, L-T (M2EB11AB1 ) WITH NEGATIVE R VALUE",
        {"fit_type": "SPLINE", "alloy": "2024", "temper": "T351", "form": "PLATE",
         "orientation": "L-T", "nasgro_code": "M2EB11AB1", "r_sign": "NEGATIVE"},
    ),
]


@pytest.mark.parametrize("title,expected", REAL_TITLE_VARIANTS)
def test_parse_title_variants(title, expected):
    parsed = parse_title(title)
    assert parsed is not None, title
    for key, value in expected.items():
        assert parsed[key] == value, f"{key}: {parsed[key]!r} != {value!r}"


def test_unparseable_title_returns_none():
    assert parse_title("SOME UNRELATED CHART") is None
