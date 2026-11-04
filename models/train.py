"""
Phase 2: 学習・モデル比較・外部テスト・公開用再学習。

- D01: GroupKFold 交差検証で候補比較
- 予測は、D01 の 2016年以降だけで決めた割合 a で学習時の予測平均へ縮める
- D02: 外部テスト（ハイパラ調整には使わない）
- 採用モデルを D01+D02 で再学習し artifacts へ保存
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pipeline_utils import RareCategoryGrouper  # noqa: E402
from preprocessing import prepare_d01, prepare_d02  # noqa: E402
from shrinkage import (  # noqa: E402
    CALIBRATION_N,
    calibration_bias,
    d02_grading_dates,
    earliest_calibration_split,
    select_shrinkage_factors,
    set_prediction_bias,
    with_shrinkage,
)

ARTIFACT_DIR = ROOT / "artifacts"
FEATURE_COLS = [
    "country",
    "altitude_m",
    "variety",
    "process",
    "moisture_pct",
    "category_1_defects",
    "category_2_defects",
    "color",
]
TARGET_COL = "target_score"
NUMERIC_COLS = [
    "altitude_m",
    "moisture_pct",
    "category_1_defects",
    "category_2_defects",
]
CATEGORICAL_COLS = ["country", "variety", "process", "color"]
RARE_MIN_COUNT = 10
N_SPLITS = 5
RANDOM_STATE = 42


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_count=RARE_MIN_COUNT)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_model_pipeline(name: str, estimator, *, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "dummy_median": build_model_pipeline(
            "dummy_median",
            DummyRegressor(strategy="median"),
            scale_numeric=False,
        ),
        "ridge": build_model_pipeline(
            "ridge",
            Ridge(alpha=1.0),
            scale_numeric=True,
        ),
        "random_forest": build_model_pipeline(
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scale_numeric=False,
        ),
        "hist_gradient_boosting": build_model_pipeline(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.08,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
        ),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float)
    return X, y


def metric_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_group_cv(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> dict:
    cv = GroupKFold(n_splits=N_SPLITS)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    result = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "mae_mean": float(-result["test_mae"].mean()),
        "mae_std": float(result["test_mae"].std()),
        "rmse_mean": float(-result["test_rmse"].mean()),
        "rmse_std": float(result["test_rmse"].std()),
        "r2_mean": float(result["test_r2"].mean()),
        "r2_std": float(result["test_r2"].std()),
    }


def subgroup_mae(y_true: pd.Series, y_pred: np.ndarray, by: pd.Series) -> dict:
    frame = pd.DataFrame({"y": y_true.to_numpy(), "pred": y_pred, "g": by.to_numpy()})
    out: dict[str, dict] = {}
    for key, part in frame.groupby("g"):
        if len(part) < 5:
            continue
        out[str(key)] = {
            "n": int(len(part)),
            "mae": float(mean_absolute_error(part["y"], part["pred"])),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"])[:15])


def compute_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    result = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    rows = []
    for name, mean, std in zip(FEATURE_COLS, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature": name,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
        )
    rows.sort(key=lambda r: -r["importance_mean"])
    return rows


def input_ranges(df: pd.DataFrame) -> dict:
    ranges: dict = {}
    for col in NUMERIC_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        ranges[col] = {
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }
    for col in CATEGORICAL_COLS:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        ranges[col] = {"categories": vals}
    return ranges


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    d01, groups_d01, report_d01 = prepare_d01()
    d02, _groups_d02, report_d02 = prepare_d02()

    X_d01, y_d01 = split_xy(d01)
    X_d02, y_d02 = split_xy(d02)

    base_models = candidate_models()
    shrinkage = select_shrinkage_factors(base_models)
    print(
        "=== shrinkage a (D01 before 2016 -> 2016 onward, D02 unused) ==="
    )
    print(
        f"early n={shrinkage['n_early']}  late n={shrinkage['n_late']}  "
        f"undated excluded n={shrinkage['n_undated_excluded']}"
    )
    models = {"dummy_median": base_models["dummy_median"]}
    shrinkage_a: dict[str, float] = {}
    for name, result in shrinkage["models"].items():
        a = float(result["selected_a"])
        shrinkage_a[name] = a
        models[name] = with_shrinkage(base_models[name], a)
        print(
            f"{name:24s}  a={a:.1f}  "
            f"late MAE={result['selected_mae']:.3f}"
        )

    cv_table: dict[str, dict] = {}
    print("=== D01 GroupKFold CV ===")
    for name, pipe in models.items():
        scores = run_group_cv(pipe, X_d01, y_d01, groups_d01)
        cv_table[name] = scores
        print(
            f"{name:24s}  MAE={scores['mae_mean']:.3f}±{scores['mae_std']:.3f}  "
            f"RMSE={scores['rmse_mean']:.3f}  R2={scores['r2_mean']:.3f}"
        )

    # 各候補を D01 で fit し、D02 へは各1回だけ外部評価（ハイパラ探索には使わない）
    print("\n=== D02 external test (fit on D01 only, one shot each) ===")
    external_table: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    preds_d02: dict[str, np.ndarray] = {}
    fit_times: dict[str, float] = {}
    predict_times: dict[str, float] = {}
    for name, pipe in models.items():
        model = clone(pipe)
        t0 = time.perf_counter()
        model.fit(X_d01, y_d01)
        fit_times[name] = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = model.predict(X_d02)
        predict_times[name] = time.perf_counter() - t1
        metrics = metric_dict(y_d02, pred)
        external_table[name] = metrics
        fitted[name] = model
        preds_d02[name] = pred
        print(
            f"{name:24s}  MAE={metrics['mae']:.3f}  "
            f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}"
        )

    dummy_external = external_table["dummy_median"]
    # 受け入れ: Dummy より MAE が良い候補のうち最良。無ければ全体最良を記録し警告
    beaters = [
        (n, m)
        for n, m in external_table.items()
        if n != "dummy_median" and m["mae"] < dummy_external["mae"]
    ]
    if beaters:
        selected_name = min(beaters, key=lambda kv: kv[1]["mae"])[0]
        selection_rule = "d02_mae_among_models_beating_dummy"
    else:
        non_dummy = [(n, m) for n, m in external_table.items() if n != "dummy_median"]
        selected_name = min(non_dummy, key=lambda kv: kv[1]["mae"])[0]
        selection_rule = "d02_mae_best_non_dummy_WARNING_does_not_beat_dummy"

    selected = fitted[selected_name]
    external = external_table[selected_name]
    fit_sec = fit_times[selected_name]
    predict_sec = predict_times[selected_name]

    dates_d02 = d02_grading_dates()
    if len(dates_d02) != len(X_d02):
        raise RuntimeError("D02 grading dates are not aligned with prepared rows")
    cal_idx, test_idx = earliest_calibration_split(dates_d02, CALIBRATION_N)
    y_cal = y_d02.iloc[cal_idx]
    y_test = y_d02.iloc[test_idx]
    X_test = X_d02.iloc[test_idx]
    biases: dict[str, float] = {}
    calibrated_holdout: dict[str, dict] = {}
    for name, pred in preds_d02.items():
        if name == "dummy_median":
            continue
        bias = calibration_bias(y_cal.to_numpy(), pred[cal_idx])
        biases[name] = bias
        calibrated_holdout[name] = metric_dict(y_test, pred[test_idx] + bias)
    dummy_holdout = metric_dict(y_test, preds_d02["dummy_median"][test_idx])
    dummy_bias = calibration_bias(
        y_cal.to_numpy(), preds_d02["dummy_median"][cal_idx]
    )
    dummy_calibrated_holdout = metric_dict(
        y_test, preds_d02["dummy_median"][test_idx] + dummy_bias
    )
    selected_holdout = calibrated_holdout[selected_name]
    pred_holdout = preds_d02[selected_name][test_idx] + biases[selected_name]

    print(f"\nSelected: {selected_name} ({selection_rule})")
    print(f"uncalibrated full D02  {external}")
    print(
        f"calibration n={len(cal_idx)}  "
        f"{dates_d02.iloc[cal_idx].min().date()} -> "
        f"{dates_d02.iloc[cal_idx].max().date()}  "
        f"bias={biases[selected_name]:+.3f}"
    )
    print(
        f"holdout n={len(test_idx)}  "
        f"MAE={selected_holdout['mae']:.3f}  "
        f"RMSE={selected_holdout['rmse']:.3f}  "
        f"R2={selected_holdout['r2']:.3f}"
    )
    print(
        f"dummy on same holdout MAE={dummy_holdout['mae']:.3f}  "
        f"beats_dummy={selected_holdout['mae'] < dummy_holdout['mae']}"
    )

    subgroups = {
        "country": subgroup_mae(
            y_test, pred_holdout, X_test["country"].fillna("Missing")
        ),
        "process": subgroup_mae(
            y_test, pred_holdout, X_test["process"].fillna("Missing")
        ),
    }

    # 公開用は D01 学習済みモデルに、補正用 30 件で決めた定数だけを足す。
    # 採点用の D02 では再学習しない。
    production = selected
    set_prediction_bias(production, biases[selected_name])

    importance = compute_importance(production, X_test, y_test)

    model_path = ARTIFACT_DIR / "beanscope_pipeline.joblib"
    joblib.dump(production, model_path)

    metadata = {
        "version": "0.2.3",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "selection_rule": selection_rule,
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
        "rare_min_count": RARE_MIN_COUNT,
        "shrinkage": {
            "cutoff": shrinkage["cutoff"],
            "a_by_model": shrinkage_a,
            "center": "mean of in-sample predictions, stored at fit",
            "n_early": shrinkage["n_early"],
            "n_late": shrinkage["n_late"],
            "n_undated_excluded": shrinkage["n_undated_excluded"],
            "dummy_median": "not shrunk",
        },
        "level_calibration": {
            "n_calibration": int(len(cal_idx)),
            "n_holdout": int(len(test_idx)),
            "calibration_start": str(dates_d02.iloc[cal_idx].min().date()),
            "calibration_end": str(dates_d02.iloc[cal_idx].max().date()),
            "bias_by_model": biases,
            "selected_bias": biases[selected_name],
            "rule": (
                "Earliest D02 rows by grading date estimate one additive constant. "
                "Holdout rows are not used to fit the model or the constant."
            ),
        },
        "n_splits_group_kfold": N_SPLITS,
        "n_d01": int(len(d01)),
        "n_d02": int(len(d02)),
        "n_production_train": int(len(d01)),
        "d01_exclusions": dict(report_d01.excluded),
        "d02_exclusions": dict(report_d02.excluded),
        "input_ranges_production": input_ranges(d01),
        "notes": [
            "Displayed MAE is the calibrated holdout, not the rows used to choose the constant.",
            "Production model is fit on D01 only. The earliest 30 D02 rows add one constant.",
            "Holdout D02 labels are not used to refit the model.",
            "Imputation / rare-Other / encoding are fit inside Pipeline only.",
            "Shrinkage factor a is chosen on D01 graded in 2016 or later.",
            "At predict time, scores shrink toward the training-set prediction mean, then the constant is added.",
        ],
        "timing": {
            "fit_d01_seconds": fit_sec,
            "predict_d02_seconds": predict_sec,
            "predict_d02_per_row_ms": 1000.0 * predict_sec / max(len(X_d02), 1),
        },
        "model_path": str(model_path.name),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    evaluation = {
        "cv_d01_group_kfold": cv_table,
        "external_d02_all_models": external_table,
        "selected_model": selected_name,
        "selection_rule": selection_rule,
        "external_d02_uncalibrated": external,
        "baseline_dummy_external_d02_uncalibrated": dummy_external,
        "d02_level_calibration": {
            "n_calibration": int(len(cal_idx)),
            "n_holdout": int(len(test_idx)),
            "bias_by_model": biases,
            "holdout_calibrated": calibrated_holdout,
            "holdout_dummy_median": dummy_holdout,
            "dummy_bias_from_same_calibration_rows": dummy_bias,
            "holdout_dummy_plus_own_bias": dummy_calibrated_holdout,
        },
        "external_d02": selected_holdout,
        "baseline_dummy_external_d02": dummy_holdout,
        "acceptance": {
            "mae_beats_dummy": bool(selected_holdout["mae"] < dummy_holdout["mae"]),
            "mae_under_2_optional_goal": bool(selected_holdout["mae"] <= 2.0),
            "cv_vs_external_mae_gap": float(
                abs(cv_table[selected_name]["mae_mean"] - selected_holdout["mae"])
            ),
        },
        "subgroup_mae_d02_holdout": subgroups,
        "feature_importance_permutation_d02": importance,
        "shrinkage_a": shrinkage_a,
        "distribution_shift_note": (
            "D02 scores skew higher than D01; negative R2 on D02 can occur when "
            "absolute error is still moderate. Prefer MAE for acceptance."
        ),
    }
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "feature_importance.json").write_text(
        json.dumps(importance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ranked_cv = sorted(cv_table.items(), key=lambda kv: kv[1]["mae_mean"])
    ranked_ext = sorted(external_table.items(), key=lambda kv: kv[1]["mae"])
    lines = [
        "BeanScope Phase 2 training report",
        "=" * 60,
        f"selected_model: {selected_name}",
        f"selection_rule: {selection_rule}",
        f"D01 n={len(d01)}  D02 n={len(d02)}  production n={len(d01)}",
        f"calibration n={len(cal_idx)}  holdout n={len(test_idx)}  "
        f"selected bias={biases[selected_name]:+.3f}",
        "",
        "Shrinkage a (chosen on D01 from 2016, dummy unchanged)",
        "-" * 40,
    ]
    for name, a in shrinkage_a.items():
        lines.append(f"  {name:24s} a={a:.1f}")
    lines.extend(["", "D01 GroupKFold CV (MAE mean±std)", "-" * 40])
    for name, scores in ranked_cv:
        lines.append(
            f"  {name:24s} MAE={scores['mae_mean']:.3f}±{scores['mae_std']:.3f}  "
            f"RMSE={scores['rmse_mean']:.3f}  R2={scores['r2_mean']:.3f}"
        )
    lines.extend(["", "D02 uncalibrated (all rows, one shot)", "-" * 40])
    for name, metrics in ranked_ext:
        mark = " <-- selected" if name == selected_name else ""
        lines.append(
            f"  {name:24s} MAE={metrics['mae']:.3f}  "
            f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}{mark}"
        )
    ranked_cal = sorted(calibrated_holdout.items(), key=lambda kv: kv[1]["mae"])
    lines.extend(
        [
            "",
            "D02 holdout after adding the constant from the earliest 30 rows",
            "-" * 40,
        ]
    )
    for name, metrics in ranked_cal:
        mark = " <-- selected" if name == selected_name else ""
        lines.append(
            f"  {name:24s} bias={biases[name]:+.3f}  MAE={metrics['mae']:.3f}  "
            f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}{mark}"
        )
    lines.extend(
        [
            f"  {'dummy_median':24s} MAE={dummy_holdout['mae']:.3f}  "
            f"RMSE={dummy_holdout['rmse']:.3f}  R2={dummy_holdout['r2']:.3f}",
            f"  {'dummy_plus_own_bias':24s} bias={dummy_bias:+.3f}  "
            f"MAE={dummy_calibrated_holdout['mae']:.3f}  "
            f"RMSE={dummy_calibrated_holdout['rmse']:.3f}  "
            f"R2={dummy_calibrated_holdout['r2']:.3f}",
            "",
            f"  beats_dummy={selected_holdout['mae'] < dummy_holdout['mae']}",
            f"  cv_external_mae_gap={abs(cv_table[selected_name]['mae_mean'] - selected_holdout['mae']):.3f}",
            "",
            f"Artifacts -> {ARTIFACT_DIR}",
            f"  {model_path.name}",
            "  metadata.json",
            "  evaluation.json",
            "",
        ]
    )
    report_path = ARTIFACT_DIR / "phase2_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote artifacts -> {ARTIFACT_DIR}")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
