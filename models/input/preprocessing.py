"""
D01 / D02 の決定的な整形スクリプト。

欠損補完・標準化・低頻度 Other 化・エンコーディングは行わない。
それらは学習 Pipeline（交差検証の各 fold の学習データ）で fit する。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH_D01 = BASE_DIR / "before" / "coffee_ratings.csv"
INPUT_PATH_D02 = BASE_DIR / "before" / "df_arabica_clean.csv"
OUTPUT_DIR = BASE_DIR / "after"
OUTPUT_CSV_D01 = OUTPUT_DIR / "coffee_ratings_clean.csv"
OUTPUT_CSV_D02 = OUTPUT_DIR / "df_arabica_clean_clean.csv"
OUTPUT_REPORT_D01 = OUTPUT_DIR / "preprocessing_report_d01.txt"
OUTPUT_REPORT_D02 = OUTPUT_DIR / "preprocessing_report_d02.txt"
# 後方互換
INPUT_PATH = INPUT_PATH_D01
OUTPUT_CSV = OUTPUT_CSV_D01
OUTPUT_REPORT = OUTPUT_REPORT_D01

OUTPUT_COLUMNS = [
    "country",
    "altitude_m",
    "variety",
    "process",
    "moisture_pct",
    "category_1_defects",
    "category_2_defects",
    "color",
    "target_score",
]

ALT_MIN_M = 200.0
ALT_MAX_M = 2500.0

# CSV 上の欠損印（文字列）。moisture==0 は含めない。
NULL_TOKENS = ["", "NA", "N/A", "None", "none", "nan", "NaN", "null", "NULL", "nil"]

# ---------------------------------------------------------------------------
# Explicit synonym / spelling maps (要件 4.4)
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    "tanzania, united republic of": "Tanzania",
    "cote d?ivoire": "Cote d'Ivoire",
    "cote d'ivoire": "Cote d'Ivoire",
    "côte d'ivoire": "Cote d'Ivoire",
}

VARIETY_MAP = {
    "gesha": "Gesha",
    "geisha": "Gesha",
    "marigojipe": "Maragogipe",
    "maragogype": "Maragogipe",
    "maragogipe": "Maragogipe",
    "pache comun": "Pache",
    "pache": "Pache",
    "ethiopian heirlooms": "Ethiopian Heirloom",
    "ethiopian heirloom": "Ethiopian Heirloom",
    "ethiopian yirgacheffe": "Yirgacheffe",
    "yirgacheffe": "Yirgacheffe",
    "sumatra lintong": "Sumatra",
    "sumatra": "Sumatra",
    "mandheling": "Mandheling",
    "yellow bourbon": "Yellow Bourbon",
    "red bourbon": "Bourbon",
    "hawaiian kona": "Hawaiian Kona",
    "mundo novo": "Mundo Novo",
    "blue mountain": "Blue Mountain",
    "ruiru 11": "Ruiru 11",
    "moka peaberry": "Moka Peaberry",
    "peaberry": "Peaberry",
    "caturra": "Caturra",
    "bourbon": "Bourbon",
    "typica": "Typica",
    "catuai": "Catuai",
    "catimor": "Catimor",
    "castillo": "Castillo",
    "pacas": "Pacas",
    "pacamara": "Pacamara",
    "arusha": "Arusha",
    "java": "Java",
    "sulawesi": "Sulawesi",
    "sl14": "SL14",
    "sl28": "SL28",
    "sl34": "SL34",
    "unknown": "Other",
    "other": "Other",
}

PROCESS_MAP = {
    "washed / wet": "Washed",
    "washed": "Washed",
    "wet": "Washed",
    "double anaerobic washed": "Washed",
    "natural / dry": "Natural",
    "natural": "Natural",
    "dry": "Natural",
    "double carbonic maceration / natural": "Natural",
    "semi-washed / semi-pulped": "Semi-Washed",
    "semi-washed": "Semi-Washed",
    "semi washed": "Semi-Washed",
    "semi-pulped": "Semi-Washed",
    "semi-lavado": "Semi-Washed",
    "pulped natural / honey": "Honey",
    "pulped natural": "Honey",
    "honey": "Honey",
    "honey,mossto": "Honey",
    "wet hulling": "Other",
    "anaerobico 1000h": "Other",
    "other": "Other",
}

COLOR_MAP = {
    "green": "Green",
    "greenish": "Green",
    "bluish-green": "Blue-Green",
    "blue-green": "Blue-Green",
    "bluish green": "Blue-Green",
    "blue green": "Blue-Green",
    "yellow-green": "Yellow-Green",
    "yellow green": "Yellow-Green",
    "yellow- green": "Yellow-Green",
    "yello-green": "Yellow-Green",
    "yellowish": "Yellow-Green",
    "pale yellow": "Yellow-Green",
    "brownish": "Brownish",
    "browish-green": "Brownish",
}


@dataclass
class Report:
    n_input: int = 0
    n_output: int = 0
    excluded: Counter = field(default_factory=Counter)
    altitude_corrections: Counter = field(default_factory=Counter)
    altitude_out_of_range_to_missing: int = 0
    notes: list[str] = field(default_factory=list)

    def add_exclusion(self, reason: str, count: int = 1) -> None:
        self.excluded[reason] += count


def is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {t.lower() for t in NULL_TOKENS}


def to_float(value: Any) -> float | None:
    if is_null(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def normalize_text(value: Any) -> str | None:
    if is_null(value):
        return None
    return re.sub(r"\s+", " ", str(value).strip())


def map_category(value: Any, mapping: dict[str, str]) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    key = text.lower()
    if key in mapping:
        return mapping[key]
    # 未登録は表記を整えて残す（低頻度 Other 化はしない）
    return text.title() if not text.isupper() else text


def normalize_country(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    key = text.lower()
    if key in COUNTRY_MAP:
        return COUNTRY_MAP[key]
    return text


def parse_altitude_from_raw(raw: Any) -> float | None:
    """生の altitude 文字列からメートル値を推定できる場合に返す。"""
    if is_null(raw):
        return None

    s = str(raw).strip()

    # 12oo → 1200
    s = re.sub(r"(?i)(\d)oo\b", r"\g<1>00", s)

    # 1'500 → 1500（千の位区切り）
    s = re.sub(r"(\d)'(\d{3})\b", r"\1\2", s)

    # "1800 meters (5900" → 1800（括弧前のメートル表記を優先）
    m = re.match(
        r"(?i)^\s*(\d+(?:\.\d+)?)\s*(?:meters?|metres?)\s*\(\d+",
        s,
    )
    if m:
        return float(m.group(1))

    # "1100.00 mosl" / "1901.64" / "1500 meters" など単純形
    m = re.match(
        r"(?i)^\s*(\d+(?:\.\d+)?)\s*"
        r"(?:mosl|masl|msnm|m\.?s\.?n\.?m\.?|m\.?s\.?l\.?|"
        r"meters?|metres?|mts?|公尺)?\s*$",
        s,
    )
    if m:
        return float(m.group(1))

    # "approx. 1500 metres a.s.l." など前置詞付き
    m = re.search(
        r"(?i)(\d+(?:\.\d+)?)\s*"
        r"(?:mosl|masl|msnm|m\.?s\.?n\.?m\.?|m\.?s\.?l\.?|"
        r"meters?|metres?|mts?|公尺)",
        s,
    )
    if m:
        return float(m.group(1))

    return None


def resolve_altitude(
    mean_raw: Any,
    altitude_text: Any,
    unit: Any,
    report: Report,
) -> float | None:
    """
    altitude_mean_meters を基準にし、生テキストから直せるものは補正する。
    最後に 200〜2500m の範囲外は欠損とする。
    """
    mean = to_float(mean_raw)
    raw_parsed = parse_altitude_from_raw(altitude_text)
    unit_norm = str(unit).strip().lower() if not is_null(unit) else "m"

    value = mean
    correction = None

    # 千の位アポストロフィ: mean が範囲内でも誤パースされていることがある
    if not is_null(altitude_text) and re.search(r"\d'\d{3}", str(altitude_text)):
        if raw_parsed is not None and ALT_MIN_M <= raw_parsed <= ALT_MAX_M:
            if mean is None or abs(mean - raw_parsed) > 1e-6:
                value, correction = raw_parsed, "apostrophe_thousands"

    # "1800 meters (5900"
    if (
        correction is None
        and not is_null(altitude_text)
        and re.search(r"(?i)meters?\s*\(", str(altitude_text))
    ):
        if raw_parsed is not None and ALT_MIN_M <= raw_parsed <= ALT_MAX_M:
            value, correction = raw_parsed, "meters_before_paren"

    # oo タイポ（12oo など）
    if (
        correction is None
        and not is_null(altitude_text)
        and re.search(r"(?i)\doo\b", str(altitude_text))
    ):
        if raw_parsed is not None and ALT_MIN_M <= raw_parsed <= ALT_MAX_M:
            value, correction = raw_parsed, "oo_typo"

    # 小数点脱落（1901.64 → 190164、1100.00 → 110000）
    if correction is None and mean is not None and raw_parsed is not None:
        if mean > ALT_MAX_M and ALT_MIN_M <= raw_parsed <= ALT_MAX_M:
            if abs(mean - raw_parsed * 100) < 1.0 or abs(mean - raw_parsed * 10) < 1.0:
                value, correction = raw_parsed, "decimal_point_dropped"

    # mean が巨大値で raw も同様（190164 / 110000）→ 桁補正
    if correction is None and mean is not None and mean >= 10000:
        for divisor in (100, 10):
            candidate = mean / divisor
            if ALT_MIN_M <= candidate <= ALT_MAX_M:
                if (
                    raw_parsed is None
                    or abs(raw_parsed - candidate) < 1.0
                    or raw_parsed == mean
                ):
                    value, correction = candidate, f"scale_div_{divisor}"
                    break

    # mean が範囲外で raw から妥当値が取れる場合
    if correction is None and mean is not None and not (ALT_MIN_M <= mean <= ALT_MAX_M):
        if raw_parsed is not None:
            candidate = raw_parsed
            if unit_norm == "ft" and candidate > ALT_MAX_M:
                candidate = candidate * 0.3048
            if ALT_MIN_M <= candidate <= ALT_MAX_M:
                value, correction = candidate, "raw_override_out_of_range_mean"

    # mean 欠損時は raw から（ft なら換算）
    if value is None and raw_parsed is not None:
        candidate = raw_parsed
        if unit_norm == "ft":
            candidate = candidate * 0.3048
        value = candidate
        correction = "from_raw_missing_mean"

    if correction:
        report.altitude_corrections[correction] += 1

    if value is None:
        return np.nan

    if not (ALT_MIN_M <= value <= ALT_MAX_M):
        report.altitude_out_of_range_to_missing += 1
        return np.nan

    return round(value, 4)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=NULL_TOKENS,
    )


def load_source() -> pd.DataFrame:
    return load_csv(INPUT_PATH_D01)


def make_group_id(
    country: pd.Series,
    farm: pd.Series,
    lot: pd.Series,
    grading_date: pd.Series,
) -> pd.Series:
    """類似ロットを同じグループにまとめるキー（学習分割用・特徴量ではない）。"""

    def _part(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().str.lower()

    return (
        _part(country)
        + "|"
        + _part(farm)
        + "|"
        + _part(lot)
        + "|"
        + _part(grading_date)
    )


def parse_d02_altitude(text: Any, report: Report) -> float:
    """D02 の Altitude（範囲表記可）をメートル中央値へ。範囲外は欠損。"""
    if is_null(text):
        return np.nan
    s = str(text).strip()
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return np.nan
    value = nums[0] if len(nums) == 1 else float(np.mean(nums[:2]))
    if not (ALT_MIN_M <= value <= ALT_MAX_M):
        report.altitude_out_of_range_to_missing += 1
        return np.nan
    return round(value, 4)


def apply_exclusions_d01(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """D01 行除外。"""
    species = df["species"].astype("string").str.strip().str.lower()
    robusta_mask = species == "robusta"
    n_robusta = int(robusta_mask.sum())
    if n_robusta:
        report.add_exclusion("robusta", n_robusta)

    target = pd.to_numeric(df["total_cup_points"], errors="coerce")
    zero_mask = target == 0
    n_zero = int(zero_mask.sum())
    if n_zero:
        for country in df.loc[zero_mask, "country_of_origin"]:
            report.notes.append(
                "Excluded 1 row with total_cup_points==0 "
                f"(country={country!r}, cupping sub-scores all zero; Honduras)."
            )
        report.add_exclusion("zero_target_score", n_zero)

    missing_target_mask = target.isna() & ~robusta_mask & ~zero_mask
    n_missing_target = int(missing_target_mask.sum())
    if n_missing_target:
        report.add_exclusion("missing_target_score", n_missing_target)

    country_missing_mask = df["country_of_origin"].isna()
    country_only = (
        country_missing_mask & ~robusta_mask & ~zero_mask & ~missing_target_mask
    )
    n_missing_country = int(country_only.sum())
    if n_missing_country:
        report.add_exclusion("missing_country", n_missing_country)

    keep = ~robusta_mask & ~zero_mask & ~missing_target_mask & ~country_missing_mask
    return df.loc[keep].copy()


def apply_exclusions(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    return apply_exclusions_d01(df, report)


def apply_exclusions_d02(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """D02 行除外（アラビカ前提ファイル）。"""
    target = pd.to_numeric(df["Total Cup Points"], errors="coerce")
    zero_mask = target == 0
    n_zero = int(zero_mask.sum())
    if n_zero:
        report.add_exclusion("zero_target_score", n_zero)

    missing_target_mask = target.isna() & ~zero_mask
    n_missing_target = int(missing_target_mask.sum())
    if n_missing_target:
        report.add_exclusion("missing_target_score", n_missing_target)

    country_missing_mask = df["Country of Origin"].isna()
    country_only = country_missing_mask & ~zero_mask & ~missing_target_mask
    n_missing_country = int(country_only.sum())
    if n_missing_country:
        report.add_exclusion("missing_country", n_missing_country)

    keep = ~zero_mask & ~missing_target_mask & ~country_missing_mask
    return df.loc[keep].copy()


def build_cleaned_frame_d01(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    altitude_m = [
        resolve_altitude(mean, raw, unit, report)
        for mean, raw, unit in zip(
            df["altitude_mean_meters"],
            df["altitude"],
            df["unit_of_measurement"],
            strict=True,
        )
    ]

    moisture = pd.to_numeric(df["moisture"], errors="coerce")
    moisture_pct = moisture * 100.0

    cleaned = pd.DataFrame(
        {
            "country": df["country_of_origin"].map(normalize_country),
            "altitude_m": altitude_m,
            "variety": df["variety"].map(lambda v: map_category(v, VARIETY_MAP)),
            "process": df["processing_method"].map(
                lambda v: map_category(v, PROCESS_MAP)
            ),
            "moisture_pct": moisture_pct,
            "category_1_defects": pd.to_numeric(
                df["category_one_defects"], errors="coerce"
            )
            .round()
            .astype("Int64"),
            "category_2_defects": pd.to_numeric(
                df["category_two_defects"], errors="coerce"
            )
            .round()
            .astype("Int64"),
            "color": df["color"].map(lambda v: map_category(v, COLOR_MAP)),
            "target_score": pd.to_numeric(df["total_cup_points"], errors="coerce"),
        }
    )
    return cleaned[OUTPUT_COLUMNS]


def build_cleaned_frame(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    return build_cleaned_frame_d01(df, report)


def build_cleaned_frame_d02(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    altitude_m = [parse_d02_altitude(v, report) for v in df["Altitude"]]
    # D02 は既に百分率
    moisture_pct = pd.to_numeric(df["Moisture Percentage"], errors="coerce")

    cleaned = pd.DataFrame(
        {
            "country": df["Country of Origin"].map(normalize_country),
            "altitude_m": altitude_m,
            "variety": df["Variety"].map(lambda v: map_category(v, VARIETY_MAP)),
            "process": df["Processing Method"].map(
                lambda v: map_category(v, PROCESS_MAP)
            ),
            "moisture_pct": moisture_pct,
            "category_1_defects": pd.to_numeric(
                df["Category One Defects"], errors="coerce"
            )
            .round()
            .astype("Int64"),
            "category_2_defects": pd.to_numeric(
                df["Category Two Defects"], errors="coerce"
            )
            .round()
            .astype("Int64"),
            "color": df["Color"].map(lambda v: map_category(v, COLOR_MAP)),
            "target_score": pd.to_numeric(df["Total Cup Points"], errors="coerce"),
        }
    )
    return cleaned[OUTPUT_COLUMNS]


def prepare_d01() -> tuple[pd.DataFrame, pd.Series, Report]:
    """整形済み D01 と分割用 group_id を返す。"""
    source = load_csv(INPUT_PATH_D01)
    report = Report(n_input=len(source))
    filtered = apply_exclusions_d01(source, report)
    cleaned = build_cleaned_frame_d01(filtered, report)
    groups = make_group_id(
        filtered["country_of_origin"],
        filtered["farm_name"],
        filtered["lot_number"],
        filtered["grading_date"],
    ).reset_index(drop=True)
    cleaned = cleaned.reset_index(drop=True)
    report.n_output = len(cleaned)
    return cleaned, groups, report


def prepare_d02() -> tuple[pd.DataFrame, pd.Series, Report]:
    """整形済み D02 と分割用 group_id を返す。"""
    source = load_csv(INPUT_PATH_D02)
    report = Report(n_input=len(source))
    filtered = apply_exclusions_d02(source, report)
    cleaned = build_cleaned_frame_d02(filtered, report)
    groups = make_group_id(
        filtered["Country of Origin"],
        filtered["Farm Name"],
        filtered["Lot Number"],
        filtered["Grading Date"],
    ).reset_index(drop=True)
    cleaned = cleaned.reset_index(drop=True)
    report.n_output = len(cleaned)
    return cleaned, groups, report


def write_report(
    report: Report,
    cleaned: pd.DataFrame,
    *,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"BeanScope {dataset_name} preprocessing report")
    lines.append("=" * 60)
    lines.append(f"Input : {input_path}")
    lines.append(f"Output: {output_path}")
    lines.append("")
    lines.append("Counts")
    lines.append("-" * 40)
    lines.append(f"rows_in                 : {report.n_input}")
    lines.append(f"rows_out                : {report.n_output}")
    lines.append(f"rows_excluded           : {report.n_input - report.n_output}")
    lines.append("")
    lines.append("Exclusions (by reason)")
    lines.append("-" * 40)
    if report.excluded:
        for reason, count in report.excluded.most_common():
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Altitude corrections (before range filter)")
    lines.append("-" * 40)
    if report.altitude_corrections:
        for reason, count in report.altitude_corrections.most_common():
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  (none)")
    lines.append(
        f"altitude_out_of_range_set_to_missing: "
        f"{report.altitude_out_of_range_to_missing}"
    )
    lines.append(f"valid_altitude_range_m: [{ALT_MIN_M:g}, {ALT_MAX_M:g}]")
    lines.append("")
    lines.append("Missing rate after cleaning (%)")
    lines.append("-" * 40)
    for col in OUTPUT_COLUMNS:
        rate = 100.0 * float(cleaned[col].isna().mean()) if len(cleaned) else 0.0
        lines.append(f"  {col}: {rate:.2f}%")
    lines.append("")
    lines.append("Notes")
    lines.append("-" * 40)
    lines.append(
        "- Imputation, scaling, encoding, and low-frequency Other collapsing "
        "are NOT applied here (要件 4.4 / 5.2)."
    )
    lines.append(
        "- moisture==0 is kept as a valid value (D01: ratio*100, D02: percent as-is)."
    )
    lines.append(
        "- Sensory / farm / certification columns are dropped (要件 3.3)."
    )
    if report.notes:
        for note in report.notes:
            lines.append(f"- {note}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _run_dataset(
    *,
    name: str,
    prepare_fn,
    input_path: Path,
    output_csv: Path,
    report_path: Path,
) -> None:
    cleaned, _groups, report = prepare_fn()
    cleaned.to_csv(output_csv, index=False, na_rep="")
    write_report(
        report,
        cleaned,
        dataset_name=name,
        input_path=input_path,
        output_path=output_csv,
        report_path=report_path,
    )
    # 後方互換の旧レポート名
    if name == "D01" and report_path != OUTPUT_DIR / "preprocessing_report.txt":
        legacy = OUTPUT_DIR / "preprocessing_report.txt"
        legacy.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[{name}] Wrote {report.n_output} rows -> {output_csv}")
    print(f"[{name}] Report -> {report_path}")
    print(f"[{name}] Excluded: {dict(report.excluded)}")
    print(f"[{name}] Altitude corrections: {dict(report.altitude_corrections)}")
    print(
        f"[{name}] Altitude out-of-range -> missing: "
        f"{report.altitude_out_of_range_to_missing}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_dataset(
        name="D01",
        prepare_fn=prepare_d01,
        input_path=INPUT_PATH_D01,
        output_csv=OUTPUT_CSV_D01,
        report_path=OUTPUT_REPORT_D01,
    )
    _run_dataset(
        name="D02",
        prepare_fn=prepare_d02,
        input_path=INPUT_PATH_D02,
        output_csv=OUTPUT_CSV_D02,
        report_path=OUTPUT_REPORT_D02,
    )


if __name__ == "__main__":
    main()

