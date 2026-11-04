"""保存済みモデルの読み込み、入力検証、1件予測。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
CATEGORICAL_COLS = ("country", "variety", "process", "color")
INTEGER_COLS = ("category_1_defects", "category_2_defects")

FEATURE_LABELS = {
    "country": "生産国",
    "altitude_m": "栽培標高",
    "variety": "品種",
    "process": "精製方法",
    "moisture_pct": "水分率",
    "category_1_defects": "Category 1 欠点数",
    "category_2_defects": "Category 2 欠点数",
    "color": "生豆の色",
}

SAMPLE_VALUES = {
    "country": "Ethiopia",
    "altitude_m": 1800.0,
    "variety": "Typica",
    "process": "Washed",
    "moisture_pct": 11.0,
    "category_1_defects": 0,
    "category_2_defects": 0,
    "color": "Green",
}


class ModelLoadError(Exception):
    """利用者が読める、モデル読込失敗のメッセージ。"""


class ValidationError(Exception):
    def __init__(self, messages: list[str]):
        super().__init__("\n".join(messages))
        self.messages = messages


@dataclass
class Artifacts:
    model: object
    metadata: dict
    evaluation: dict
    importance: list[dict]
    kept_categories: dict[str, set[str]]
    imputation_values: dict[str, float | str]


@dataclass
class CheckedInput:
    frame: pd.DataFrame
    summary: list[tuple[str, str]]
    warnings: list[str]


def _ensure_unpickle_path() -> None:
    for path in (ROOT / "models", ROOT / "models" / "input"):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _imputation_values(model) -> dict[str, float | str]:
    """学習時に欠損へ入れた値。数値は中央値、カテゴリは最頻値。"""
    preprocess = model.named_steps["preprocess"]
    numeric = preprocess.named_transformers_["num"].named_steps["imputer"]
    categorical = preprocess.named_transformers_["cat"].named_steps["imputer"]
    numeric_cols = (
        "altitude_m",
        "moisture_pct",
        "category_1_defects",
        "category_2_defects",
    )
    values: dict[str, float | str] = {
        column: float(value)
        for column, value in zip(numeric_cols, numeric.statistics_, strict=True)
    }
    values.update(
        {
            column: str(value)
            for column, value in zip(CATEGORICAL_COLS, categorical.statistics_, strict=True)
        }
    )
    return values


def _kept_categories(model) -> dict[str, set[str]]:
    preprocess = model.named_steps["preprocess"]
    grouper = preprocess.named_transformers_["cat"].named_steps["rare"]
    return {
        column: set(levels)
        for column, levels in zip(CATEGORICAL_COLS, grouper.keep_levels_, strict=True)
    }


def load_artifacts() -> Artifacts:
    """モデルとメタデータの整合を確認してから読み込む。"""
    metadata_path = ARTIFACT_DIR / "metadata.json"
    evaluation_path = ARTIFACT_DIR / "evaluation.json"
    importance_path = ARTIFACT_DIR / "feature_importance.json"
    for path in (metadata_path, evaluation_path, importance_path):
        if not path.is_file():
            raise ModelLoadError("予測に必要なファイルが見つかりません。")

    metadata = _read_json(metadata_path)
    evaluation = _read_json(evaluation_path)
    importance = _read_json(importance_path)
    if not isinstance(metadata, dict) or not isinstance(evaluation, dict):
        raise ModelLoadError("予測に必要なファイルを読み込めません。")
    if not isinstance(importance, list):
        raise ModelLoadError("予測に必要なファイルを読み込めません。")

    features = metadata.get("feature_columns")
    if features != list(FEATURE_LABELS):
        raise ModelLoadError("モデルと入力項目の対応が確認できません。")
    if metadata.get("selected_model") != evaluation.get("selected_model"):
        raise ModelLoadError("モデルと評価結果の対応が確認できません。")
    holdout = evaluation.get("external_d02")
    if not isinstance(holdout, dict) or "mae" not in holdout:
        raise ModelLoadError("誤差の目安を読み込めません。")

    model_path = ARTIFACT_DIR / str(metadata.get("model_path", ""))
    if not model_path.is_file():
        raise ModelLoadError("学習済みモデルが見つかりません。")

    _ensure_unpickle_path()
    try:
        model = joblib.load(model_path)
        kept = _kept_categories(model)
        fills = _imputation_values(model)
        probe = pd.DataFrame([{key: SAMPLE_VALUES[key] for key in features}])
        score = float(np.asarray(model.predict(probe), dtype=float).reshape(-1)[0])
    except Exception as exc:
        raise ModelLoadError("学習済みモデルを読み込めません。") from exc
    if not np.isfinite(score):
        raise ModelLoadError("学習済みモデルの動作を確認できません。")

    return Artifacts(
        model=model,
        metadata=metadata,
        evaluation=evaluation,
        importance=importance,
        kept_categories=kept,
        imputation_values=fills,
    )


def categories_for(artifacts: Artifacts, column: str) -> list[str]:
    values = artifacts.metadata["input_ranges_production"][column]["categories"]
    return list(values)


def numeric_range(artifacts: Artifacts, column: str) -> tuple[float, float]:
    spec = artifacts.metadata["input_ranges_production"][column]
    return float(spec["min"]), float(spec["max"])


def _require_choice(value: object, label: str, allowed: list[str]) -> tuple[str | None, str | None]:
    if value is None or str(value).strip() == "":
        return None, f"{label}を選択してください。"
    text = str(value)
    if text not in allowed:
        return None, f"{label}は学習データにある区分から選んでください。"
    return text, None


def _require_number(value: object, label: str) -> tuple[float | None, str | None]:
    if value is None or value == "":
        return None, f"{label}を入力してください。"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{label}は数値で入力してください。"
    if not np.isfinite(number):
        return None, f"{label}は数値で入力してください。"
    if number < 0:
        return None, f"{label}に負の数は使えません。"
    return number, None


def _format_range(low: float, high: float, unit: str) -> str:
    return f"{_trim(low)}〜{_trim(high)}{unit}"


def _trim(number: float) -> str:
    if float(number).is_integer():
        return f"{int(number):,}"
    return f"{number:.1f}"


def validate_input(artifacts: Artifacts, values: dict) -> CheckedInput:
    """必須・型を検証し、範囲外と低頻度区分は警告にして通す。"""
    errors: list[str] = []
    warnings: list[str] = []
    parsed: dict[str, object] = {}
    summary: list[tuple[str, str]] = []
    ranges = artifacts.metadata["input_ranges_production"]

    for column in ("country", "variety", "process"):
        label = FEATURE_LABELS[column]
        choice, error = _require_choice(
            values.get(column), label, list(ranges[column]["categories"])
        )
        if error:
            errors.append(error)
            continue
        parsed[column] = choice
        summary.append((label, str(choice)))
        _warn_category(artifacts, column, str(choice), label, warnings)

    altitude, error = _require_number(values.get("altitude_m"), FEATURE_LABELS["altitude_m"])
    if error:
        errors.append(error)
    else:
        parsed["altitude_m"] = altitude
        summary.append((FEATURE_LABELS["altitude_m"], f"{_trim(altitude)} m"))
        low, high = numeric_range(artifacts, "altitude_m")
        if altitude < low or altitude > high:
            warnings.append(
                f"栽培標高 {_trim(altitude)} m は学習データの範囲"
                f"（{_format_range(low, high, ' m')}）の外です。"
                "予測は実行しますが、この範囲の外では誤差が大きくなりやすくなります。"
            )

    moisture, error = _require_number(values.get("moisture_pct"), FEATURE_LABELS["moisture_pct"])
    if error:
        errors.append(error)
    else:
        parsed["moisture_pct"] = moisture
        summary.append((FEATURE_LABELS["moisture_pct"], f"{moisture:.1f} %"))
        low, high = numeric_range(artifacts, "moisture_pct")
        if moisture < low or moisture > high:
            warnings.append(
                f"水分率 {moisture:.1f} % は学習データの範囲"
                f"（{_format_range(low, high, ' %')}）の外です。"
                "予測は実行しますが、この範囲の外では誤差が大きくなりやすくなります。"
            )

    for column in INTEGER_COLS:
        label = FEATURE_LABELS[column]
        number, error = _require_number(values.get(column), label)
        if error:
            errors.append(error)
            continue
        if abs(number - round(number)) > 1e-9:
            errors.append(f"{label}は整数で入力してください。")
            continue
        parsed[column] = int(round(number))
        summary.append((label, f"{int(round(number))}"))
        low, high = numeric_range(artifacts, column)
        if number < low or number > high:
            warnings.append(
                f"{label} {int(round(number))} は学習データの範囲"
                f"（{_trim(low)}〜{_trim(high)}）の外です。"
                "予測は実行しますが、この範囲の外では誤差が大きくなりやすくなります。"
            )

    color_raw = values.get("color")
    if color_raw is None or str(color_raw).strip() in {"", "未指定"}:
        parsed["color"] = np.nan
        summary.append((FEATURE_LABELS["color"], "未指定"))
        warnings.append(
            "生豆の色が未指定です。学習データでもっとも多かった色で補って予測します。"
        )
    else:
        choice, error = _require_choice(
            color_raw, FEATURE_LABELS["color"], list(ranges["color"]["categories"])
        )
        if error:
            errors.append(error)
        else:
            parsed["color"] = choice
            summary.append((FEATURE_LABELS["color"], str(choice)))
            _warn_category(artifacts, "color", str(choice), FEATURE_LABELS["color"], warnings)

    if errors:
        raise ValidationError(errors)

    frame = pd.DataFrame([{column: parsed[column] for column in FEATURE_LABELS}])
    return CheckedInput(frame=frame, summary=summary, warnings=warnings)


def _warn_category(
    artifacts: Artifacts,
    column: str,
    value: str,
    label: str,
    warnings: list[str],
) -> None:
    if value == "Other":
        warnings.append(
            f"{label}の「Other」は、出現の少ない区分をまとめたものです。"
            "個別の区分より予測の頼りは下がります。"
        )
        return
    kept = artifacts.kept_categories.get(column)
    if kept is not None and value not in kept:
        warnings.append(
            f"{label}「{value}」は学習データでの出現が少ないため、"
            "予測時は Other にまとめて扱います。個別の区分より予測の頼りは下がります。"
        )


def predict_score(artifacts: Artifacts, checked: CheckedInput) -> float:
    score = float(np.asarray(artifacts.model.predict(checked.frame), dtype=float).reshape(-1)[0])
    if not np.isfinite(score):
        raise ModelLoadError("予測結果を計算できませんでした。入力を確認してください。")
    return score
