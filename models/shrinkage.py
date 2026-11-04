"""予測を、学習時の予測平均へ割合 a で縮める。"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline

from preprocessing import (
    INPUT_PATH_D01,
    INPUT_PATH_D02,
    Report,
    apply_exclusions_d01,
    apply_exclusions_d02,
    build_cleaned_frame_d01,
    load_csv,
)

CUTOFF = pd.Timestamp("2016-01-01")
A_GRID = [round(i / 10, 1) for i in range(11)]
TUNED_MODELS = ("ridge", "random_forest", "hist_gradient_boosting")
CALIBRATION_N = 30
_ORDINAL_SUFFIX = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


class ShrunkRegressor(BaseEstimator, RegressorMixin):
    """学習データの予測平均へ、固定した割合 a で予測を縮める。"""

    def __init__(self, estimator=None, a: float = 1.0, bias: float = 0.0):
        self.estimator = estimator
        self.a = a
        self.bias = bias

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        fitted = np.asarray(self.estimator_.predict(X), dtype=float)
        self.center_ = float(np.mean(fitted))
        self.a_ = float(self.a)
        self.bias_ = float(self.bias)
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X), dtype=float)
        shrunk = self.center_ + self.a_ * (pred - self.center_)
        return shrunk + self.bias_


def dated_d01() -> tuple[pd.DataFrame, pd.Series]:
    """整形済み D01 と、同じ行順の評価日を返す。日付を解釈できない行は NaT。"""
    source = load_csv(INPUT_PATH_D01)
    report = Report(n_input=len(source))
    filtered = apply_exclusions_d01(source, report)
    dates = pd.to_datetime(filtered["grading_date"], errors="coerce").reset_index(
        drop=True
    )
    cleaned = build_cleaned_frame_d01(filtered, report).reset_index(drop=True)
    return cleaned, dates


def choose_a(pred: np.ndarray, y_true: np.ndarray) -> dict:
    center = float(np.mean(pred))
    grid = []
    for a in A_GRID:
        adjusted = center + a * (pred - center)
        grid.append({"a": a, "mae": float(np.mean(np.abs(y_true - adjusted)))})
    best = min(grid, key=lambda row: (row["mae"], row["a"]))
    return {
        "prediction_center": center,
        "grid": grid,
        "selected_a": best["a"],
        "selected_mae": best["mae"],
    }


def select_shrinkage_factors(pipelines: dict[str, Pipeline]) -> dict:
    """D02 を使わず、D01 の 2016年以降の MAE が最小の a をモデルごとに決める。"""
    from train import split_xy

    cleaned, dates = dated_d01()
    early = dates.notna() & (dates < CUTOFF)
    late = dates.notna() & (dates >= CUTOFF)
    X_early, y_early = split_xy(cleaned.loc[early])
    X_late, y_late = split_xy(cleaned.loc[late])
    y_late_np = y_late.to_numpy()
    chosen: dict[str, dict] = {
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "n_undated_excluded": int(dates.isna().sum()),
        "cutoff": str(CUTOFF.date()),
        "models": {},
    }
    for name in TUNED_MODELS:
        model = clone(pipelines[name])
        model.fit(X_early, y_early)
        chosen["models"][name] = choose_a(model.predict(X_late), y_late_np)
    return chosen


def parse_grading_dates(series: pd.Series) -> pd.Series:
    """'September 21st, 2022' のような序数付き日付を datetime にする。"""
    text = series.astype(str).str.replace(_ORDINAL_SUFFIX, r"\1", regex=True)
    return pd.to_datetime(text, errors="coerce")


def d02_grading_dates() -> pd.Series:
    """prepare_d02 と同じ行順の評価日。"""
    source = load_csv(INPUT_PATH_D02)
    report = Report(n_input=len(source))
    filtered = apply_exclusions_d02(source, report)
    return parse_grading_dates(filtered["Grading Date"]).reset_index(drop=True)


def earliest_calibration_split(
    dates: pd.Series, n_calibration: int = CALIBRATION_N
) -> tuple[np.ndarray, np.ndarray]:
    """評価日が早い順に n 件を補正用、残りを採点用にする。同日は元の行順。"""
    if int(dates.isna().sum()):
        raise ValueError("D02 grading date could not be parsed")
    if n_calibration <= 0 or n_calibration >= len(dates):
        raise ValueError("calibration size must leave at least one test row")
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    return order[:n_calibration], order[n_calibration:]


def calibration_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """正解から予測を引いた平均。予測に足す定数。"""
    return float(np.mean(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))


def set_prediction_bias(pipe: Pipeline, bias: float) -> None:
    step = pipe.named_steps["model"]
    step.bias = float(bias)
    step.bias_ = float(bias)


def with_shrinkage(pipe: Pipeline, a: float) -> Pipeline:
    cloned = clone(pipe)
    steps = []
    for name, step in cloned.steps:
        if name == "model":
            steps.append((name, ShrunkRegressor(estimator=step, a=a)))
        else:
            steps.append((name, step))
    return Pipeline(steps)
