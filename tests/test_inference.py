"""保存済みモデルの読込、入力検証、固定入力の予測。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    FEATURE_LABELS,
    SAMPLE_VALUES,
    ValidationError,
    load_artifacts,
    predict_score,
    validate_input,
)

# 初期値（Ethiopia / 1800 m / Typica / Washed / 11% / 欠点 0 / Green）の表示スコア。
EXPECTED_SAMPLE_SCORE = 84.6


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_pipeline_loads_and_matches_metadata(artifacts) -> None:
    assert artifacts.metadata["selected_model"] == "random_forest"
    assert artifacts.metadata["feature_columns"] == list(FEATURE_LABELS)
    assert "quakers" not in artifacts.metadata["feature_columns"]
    assert artifacts.metadata["model_path"] == "beanscope_pipeline.joblib"


def test_model_file_is_within_deploy_limit() -> None:
    from src.inference import ARTIFACT_DIR

    size = (ARTIFACT_DIR / "beanscope_pipeline.joblib").stat().st_size
    assert size < 50 * 1024 * 1024


def test_normal_input_builds_features_and_finite_score(artifacts) -> None:
    checked = validate_input(artifacts, SAMPLE_VALUES)
    assert list(checked.frame.columns) == list(FEATURE_LABELS)
    assert checked.warnings == []
    started = time.perf_counter()
    score = predict_score(artifacts, checked)
    elapsed = time.perf_counter() - started
    assert np.isfinite(score)
    assert round(score, 1) == EXPECTED_SAMPLE_SCORE
    assert elapsed < 2.0


def test_missing_color_is_imputed_and_still_predicts(artifacts) -> None:
    values = dict(SAMPLE_VALUES)
    values["color"] = None
    checked = validate_input(artifacts, values)
    assert pd.isna(checked.frame.loc[0, "color"])
    assert any("未指定" in message for message in checked.warnings)
    assert np.isfinite(predict_score(artifacts, checked))


def test_unknown_category_is_rejected(artifacts) -> None:
    values = dict(SAMPLE_VALUES)
    values["variety"] = "Not A Real Variety"
    with pytest.raises(ValidationError) as caught:
        validate_input(artifacts, values)
    assert any("品種" in message for message in caught.value.messages)


def test_negative_number_is_rejected(artifacts) -> None:
    values = dict(SAMPLE_VALUES)
    values["altitude_m"] = -5
    with pytest.raises(ValidationError) as caught:
        validate_input(artifacts, values)
    assert any("負の数" in message for message in caught.value.messages)


def test_out_of_range_number_warns_and_still_predicts(artifacts) -> None:
    values = dict(SAMPLE_VALUES)
    values["altitude_m"] = 100
    checked = validate_input(artifacts, values)
    assert any("範囲" in message for message in checked.warnings)
    assert np.isfinite(predict_score(artifacts, checked))


def test_rare_variety_warns_that_it_becomes_other(artifacts) -> None:
    values = dict(SAMPLE_VALUES)
    values["variety"] = "Arusha"
    checked = validate_input(artifacts, values)
    assert any("Other" in message for message in checked.warnings)


def test_training_imputation_values(artifacts) -> None:
    fills = artifacts.imputation_values
    assert fills["altitude_m"] == pytest.approx(1325.88)
    assert fills["variety"] == "Caturra"
    assert fills["process"] == "Washed"
    assert fills["color"] == "Green"
