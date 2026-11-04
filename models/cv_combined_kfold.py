"""
D01+D02 を混ぜた KFold によるモデル比較。

Phase 2 の採用判断（D01 で学習し D02 で1回評価）とは別に、
結合データの中で Ridge が Dummy より誤差が小さいかを確認する。
既存の models/artifacts 直下の成果物は更新しない。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from preprocessing import prepare_d01, prepare_d02  # noqa: E402
from train import N_SPLITS, RANDOM_STATE, candidate_models, split_xy  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "combined_kfold"


def target_summary(y: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "median": float(y.median()),
        "min": float(y.min()),
        "max": float(y.max()),
    }


def run_kfold(pipe, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    result = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    mae = -result["test_mae"]
    rmse = -result["test_rmse"]
    r2 = result["test_r2"]
    return {
        "mae_mean": float(mae.mean()),
        "mae_std": float(mae.std()),
        "rmse_mean": float(rmse.mean()),
        "rmse_std": float(rmse.std()),
        "r2_mean": float(r2.mean()),
        "r2_std": float(r2.std()),
        "folds": {
            "mae": [float(v) for v in mae],
            "rmse": [float(v) for v in rmse],
            "r2": [float(v) for v in r2],
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    d01, _, _ = prepare_d01()
    d02, _, _ = prepare_d02()
    combined = pd.concat([d01, d02], ignore_index=True)
    X, y = split_xy(combined)

    print(
        f"=== D01+D02 shuffled KFold (n_splits={N_SPLITS}, "
        f"n={len(combined)}) ==="
    )
    cv_table: dict[str, dict] = {}
    for name, pipe in candidate_models().items():
        scores = run_kfold(pipe, X, y)
        cv_table[name] = scores
        print(
            f"{name:24s}  MAE={scores['mae_mean']:.3f}±{scores['mae_std']:.3f}  "
            f"RMSE={scores['rmse_mean']:.3f}  R2={scores['r2_mean']:.3f}"
        )

    ranked = sorted(cv_table.items(), key=lambda kv: kv[1]["mae_mean"])
    best_name = ranked[0][0]
    dummy_mae = cv_table["dummy_median"]["mae_mean"]
    ridge_mae = cv_table["ridge"]["mae_mean"]

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Diagnostic KFold on pooled D01+D02. "
            "Does not replace Phase 2 selection or production artifacts."
        ),
        "cv": {
            "splitter": "KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "target_score": {
            "d01": target_summary(d01["target_score"].astype(float)),
            "d02": target_summary(d02["target_score"].astype(float)),
            "combined": target_summary(y),
        },
        "models": cv_table,
        "ranked_by_mae": [name for name, _ in ranked],
        "best_by_mae": best_name,
        "ridge_beats_dummy_mae": bool(ridge_mae < dummy_mae),
        "notes": [
            "Each fold fits imputation, rare-category grouping, and scaling on its training rows only.",
            "Shuffle mixes D01 and D02 into both train and validation.",
            "The same farm or lot can appear on both sides of a fold.",
            "Phase 2 artifacts under models/artifacts/*.json are left unchanged.",
        ],
    }
    json_path = OUTPUT_DIR / "evaluation.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "BeanScope combined KFold diagnostic",
        "=" * 60,
        "D01+D02 をシャッフルした 5-fold。Phase 2 の採用結果は変更しない。",
        f"D01 n={len(d01)}  D02 n={len(d02)}  combined n={len(combined)}",
        "",
        "target_score",
        "-" * 40,
    ]
    for label, summary in payload["target_score"].items():
        lines.append(
            f"  {label:10s} n={summary['n']:4d}  "
            f"mean={summary['mean']:.2f}  median={summary['median']:.2f}  "
            f"std={summary['std']:.2f}  "
            f"min={summary['min']:.2f}  max={summary['max']:.2f}"
        )
    lines.extend(["", "KFold CV (MAE mean±std, ranked by MAE)", "-" * 40])
    for name, scores in ranked:
        mark = " <-- lowest MAE" if name == best_name else ""
        lines.append(
            f"  {name:24s} MAE={scores['mae_mean']:.3f}±{scores['mae_std']:.3f}  "
            f"RMSE={scores['rmse_mean']:.3f}±{scores['rmse_std']:.3f}  "
            f"R2={scores['r2_mean']:.3f}±{scores['r2_std']:.3f}{mark}"
        )
    lines.extend(
        [
            "",
            f"best_by_mae: {best_name}",
            f"ridge_beats_dummy_mae: {ridge_mae < dummy_mae}",
            "",
            f"Wrote {json_path}",
            "",
        ]
    )
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
