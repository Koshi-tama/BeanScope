"""
D01 の GroupKFold だけでハイパーパラメータを探索する。

外部テストで Dummy を下回れなかったことを受けた追加試行。
D02 は探索に使わず、各モデルの最良パラメータが決まってから1回だけ予測する。
models/artifacts 直下の Phase 2 成果物は更新しない。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
)

ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from preprocessing import prepare_d01, prepare_d02  # noqa: E402
from train import (  # noqa: E402
    N_SPLITS,
    RANDOM_STATE,
    candidate_models,
    metric_dict,
    split_xy,
)

OUTPUT_DIR = ROOT / "artifacts" / "tuning"
PHASE2_EVALUATION = ROOT / "artifacts" / "evaluation.json"

RIDGE_ALPHA = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
RF_SPACE = {
    "model__n_estimators": [100, 200, 400, 600],
    "model__max_depth": [4, 8, 12, 16, None],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": [0.3, 0.5, 0.8, 1.0, "sqrt"],
}
HGB_SPACE = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "model__max_iter": [100, 200, 400],
    "model__max_depth": [3, 6, None],
    "model__min_samples_leaf": [10, 20, 40],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}
RANDOM_N_ITER = 24


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def summarize_search(search, *, kind: str, n_iter: int | None) -> dict:
    index = int(search.best_index_)
    results = search.cv_results_
    order = np.argsort(results["mean_test_score"])[::-1]
    top = []
    for rank, row in enumerate(order[:5], start=1):
        top.append(
            {
                "rank": rank,
                "mae_mean": float(-results["mean_test_score"][row]),
                "mae_std": float(results["std_test_score"][row]),
                "params": json_safe(results["params"][row]),
            }
        )
    n_candidates = int(len(results["params"]))
    return {
        "search": kind,
        "n_iter_requested": n_iter,
        "n_candidates": n_candidates,
        "n_splits": N_SPLITS,
        "n_fits": n_candidates * N_SPLITS,
        "best_params": json_safe(search.best_params_),
        "cv_mae_mean": float(-results["mean_test_score"][index]),
        "cv_mae_std": float(results["std_test_score"][index]),
        "top_candidates": top,
    }


def tune_model(name: str, pipe, X, y, groups) -> dict:
    cv = GroupKFold(n_splits=N_SPLITS)
    common = dict(
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    if name == "ridge":
        search = GridSearchCV(pipe, param_grid={"model__alpha": RIDGE_ALPHA}, **common)
        kind, n_iter = "grid", None
    elif name == "random_forest":
        search = RandomizedSearchCV(
            pipe,
            param_distributions=RF_SPACE,
            n_iter=RANDOM_N_ITER,
            random_state=RANDOM_STATE,
            **common,
        )
        kind, n_iter = "random", RANDOM_N_ITER
    elif name == "hist_gradient_boosting":
        search = RandomizedSearchCV(
            pipe,
            param_distributions=HGB_SPACE,
            n_iter=RANDOM_N_ITER,
            random_state=RANDOM_STATE,
            **common,
        )
        kind, n_iter = "random", RANDOM_N_ITER
    else:
        raise ValueError(name)

    started = time.perf_counter()
    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - started
    summary = summarize_search(search, kind=kind, n_iter=n_iter)
    summary["fit_seconds"] = elapsed
    summary["estimator"] = search.best_estimator_
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    d01, groups_d01, _ = prepare_d01()
    d02, _, _ = prepare_d02()
    X_d01, y_d01 = split_xy(d01)
    X_d02, y_d02 = split_xy(d02)
    models = candidate_models()

    dummy = models["dummy_median"]
    dummy.fit(X_d01, y_d01)
    dummy_d02 = metric_dict(y_d02, dummy.predict(X_d02))

    tuned: dict[str, dict] = {}
    for name in ("ridge", "random_forest", "hist_gradient_boosting"):
        print(f"=== tune {name} on D01 GroupKFold ===", flush=True)
        summary = tune_model(name, models[name], X_d01, y_d01, groups_d01)
        estimator = summary.pop("estimator")
        external = metric_dict(y_d02, estimator.predict(X_d02))
        summary["external_d02"] = external
        summary["beats_dummy_external_mae"] = bool(external["mae"] < dummy_d02["mae"])
        tuned[name] = summary
        print(
            f"{name:24s}  CV MAE={summary['cv_mae_mean']:.3f}  "
            f"D02 MAE={external['mae']:.3f}  "
            f"params={summary['best_params']}",
            flush=True,
        )

    selected_name = min(tuned, key=lambda name: tuned[name]["cv_mae_mean"])
    selected = tuned[selected_name]
    phase2 = json.loads(PHASE2_EVALUATION.read_text(encoding="utf-8"))
    total_fits = sum(row["n_fits"] for row in tuned.values()) + len(tuned)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": (
            "Phase 2 external test MAE did not beat dummy_median. "
            "This search was run afterwards. Hyperparameters were chosen on "
            "D01 GroupKFold only. Each tuned model was scored once on D02."
        ),
        "selection_rule": "lowest D01 GroupKFold MAE among tuned models",
        "selected_by_d01_cv": selected_name,
        "cv": {
            "splitter": "GroupKFold",
            "n_splits": N_SPLITS,
            "data": "D01",
        },
        "n_fits_including_refit": total_fits,
        "dummy_median_external_d02": dummy_d02,
        "phase2_untuned_external_d02": phase2["external_d02_all_models"],
        "models": tuned,
        "selected_external_d02": selected["external_d02"],
        "selected_beats_dummy_external_mae": selected["beats_dummy_external_mae"],
        "notes": [
            "D02 labels were not used to choose parameters or the selected model.",
            "Phase 2 artifacts under models/artifacts/*.json and the production joblib are unchanged.",
            "A one-shot D02 score can still be worse than Dummy when the score distribution shifts.",
        ],
    }
    json_path = OUTPUT_DIR / "search.json"
    json_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "BeanScope hyperparameter search",
        "=" * 60,
        "D01 GroupKFold でパラメータを選択し、D02 は各モデル1回だけ評価した。",
        "Phase 2 の成果物は変更していない。",
        f"D01 n={len(d01)}  D02 n={len(d02)}  fits={total_fits}",
        f"dummy D02 MAE={dummy_d02['mae']:.3f}  "
        f"RMSE={dummy_d02['rmse']:.3f}  R2={dummy_d02['r2']:.3f}",
        "",
        "Tuned models (ranked by D01 CV MAE)",
        "-" * 40,
    ]
    ranked = sorted(tuned.items(), key=lambda kv: kv[1]["cv_mae_mean"])
    for name, row in ranked:
        mark = " <-- selected on D01 CV" if name == selected_name else ""
        external = row["external_d02"]
        lines.append(
            f"  {name:24s} CV MAE={row['cv_mae_mean']:.3f}±{row['cv_mae_std']:.3f}  "
            f"D02 MAE={external['mae']:.3f}  RMSE={external['rmse']:.3f}  "
            f"R2={external['r2']:.3f}  "
            f"beats_dummy={row['beats_dummy_external_mae']}{mark}"
        )
        lines.append(f"    params={row['best_params']}")
        untuned = phase2["external_d02_all_models"][name]
        lines.append(
            f"    phase2 D02 MAE={untuned['mae']:.3f}  "
            f"delta={external['mae'] - untuned['mae']:+.3f}"
        )
    lines.extend(
        [
            "",
            f"selected_by_d01_cv: {selected_name}",
            f"selected_beats_dummy_external_mae: {selected['beats_dummy_external_mae']}",
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
