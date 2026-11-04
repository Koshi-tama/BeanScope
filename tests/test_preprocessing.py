"""単位変換とカテゴリ正規化。"""

from __future__ import annotations

import math

from preprocessing import (
    COLOR_MAP,
    PROCESS_MAP,
    VARIETY_MAP,
    Report,
    map_category,
    normalize_country,
    parse_altitude_from_raw,
    parse_d02_altitude,
    resolve_altitude,
)


def test_altitude_text_repairs_common_typos() -> None:
    assert parse_altitude_from_raw("12oo") == 1200
    assert parse_altitude_from_raw("1'500") == 1500
    assert parse_altitude_from_raw("1800 meters (5900") == 1800
    assert parse_altitude_from_raw("approx. 1500 metres a.s.l.") == 1500


def test_resolve_altitude_repairs_dropped_decimal_and_rejects_out_of_range() -> None:
    report = Report()
    repaired = resolve_altitude(190164, "1901.64", "m", report)
    assert repaired == 1901.64
    assert report.altitude_corrections["decimal_point_dropped"] == 1

    out_of_range = resolve_altitude(100, None, "m", report)
    assert math.isnan(out_of_range)
    assert report.altitude_out_of_range_to_missing == 1


def test_resolve_altitude_converts_feet_when_mean_is_missing() -> None:
    report = Report()
    meters = resolve_altitude(None, "5000 meters", "ft", report)
    # 単位が ft のとき、生テキストの数値をメートルへ換算する。
    assert meters == round(5000 * 0.3048, 4)


def test_d02_altitude_uses_range_midpoint_and_drops_out_of_range() -> None:
    report = Report()
    assert parse_d02_altitude("1200-1800", report) == 1500
    assert math.isnan(parse_d02_altitude("100", report))
    assert report.altitude_out_of_range_to_missing == 1


def test_category_synonyms_map_to_canonical_names() -> None:
    assert map_category("geisha", VARIETY_MAP) == "Gesha"
    assert map_category("washed / wet", PROCESS_MAP) == "Washed"
    assert map_category("greenish", COLOR_MAP) == "Green"
    assert map_category(None, VARIETY_MAP) is None
    assert normalize_country("Tanzania, United Republic of") == "Tanzania"
