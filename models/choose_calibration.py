"""
予測に足す定数を、評価日が早い D02 の 30 件だけで決めて保存する。

定数は、すでに割合 a で縮めた予測に対する「正解 - 予測」の平均。
採点用の行はこの平均に使わない。学習そのものは models/train.py が行う。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from preprocessing import prepare_d01, prepare_d02  # noqa: E402
from shrinkage import (  # noqa: E402
    CALIBRATION_N,
    calibration_bias,
    d02_grading_dates,
    earliest_calibration_split,
    select_shrinkage_factors,
    with_shrinkage,
)
from train import candidate_models, metric_dict, split_xy  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "calibration"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d01, _, _ = prepare_d01()
    d02, _, _ = prepare_d02()
    X_d01, y_d01 = split_xy(d01)
    X_d02, y_d02 = split_xy(d02)
    dates = d02_grading_dates()
    if len(dates) != len(X_d02):
        raise RuntimeError("D02 grading dates are not aligned with prepared rows")
    cal_idx, test_idx = earliest_calibration_split(dates, CALIBRATION_N)
    y_cal = y_d02.iloc[cal_idx].to_numpy()
    y_test = y_d02.iloc[test_idx]

    base_models = candidate_models()
    shrinkage = select_shrinkage_factors(base_models)
    models = {"dummy_median": base_models["dummy_median"]}
    for name, result in shrinkage["models"].items():
        models[name] = with_shrinkage(base_models[name], float(result["selected_a"]))

    chosen: dict[str, dict] = {}
    print(
        f"calibration n={len(cal_idx)}  "
        f"{dates.iloc[cal_idx].min().date()} -> {dates.iloc[cal_idx].max().date()}  "
        f"holdout n={len(test_idx)} (not used to choose the constant)"
    )
    for name, pipe in models.items():
        pipe.fit(X_d01, y_d01)
        pred = np.asarray(pipe.predict(X_d02), dtype=float)
        bias = calibration_bias(y_cal, pred[cal_idx])
        holdout = metric_dict(y_test, pred[test_idx] + bias)
        chosen[name] = {
            "a": None
            if name == "dummy_median"
            else shrinkage["models"][name]["selected_a"],
            "bias": bias,
            "calibration_prediction_mean": float(np.mean(pred[cal_idx])),
            "calibration_actual_mean": float(np.mean(y_cal)),
            "holdout_not_used_for_bias": holdout,
        }
        print(
            f"{name:24s}  a={chosen[name]['a']}  bias={bias:+.3f}  "
            f"cal pred={chosen[name]['calibration_prediction_mean']:.3f}  "
            f"cal actual={chosen[name]['calibration_actual_mean']:.3f}"
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_calibration": int(len(cal_idx)),
        "n_holdout": int(len(test_idx)),
        "calibration_start": str(dates.iloc[cal_idx].min().date()),
        "calibration_end": str(dates.iloc[cal_idx].max().date()),
        "rule": (
            "Fit on all D01. Shrink with a chosen on D01 from 2016 onward. "
            "bias = mean(actual - shrunk prediction) on the earliest "
            f"{CALIBRATION_N} D02 rows. Holdout labels are not used."
        ),
        "models": chosen,
    }
    json_path = OUTPUT_DIR / "chosen_bias.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "Level-calibration constant, chosen on the earliest 30 D02 rows",
        "=" * 60,
        "D01 全体で学習し、a で縮めた予測に対して",
        "評価日が早い D02 の 30 件で bias = mean(正解 - 予測) を決める。",
        "残り 177 件の正解はこの平均に使っていない。",
        f"calibration {payload['calibration_start']} -> {payload['calibration_end']}",
        f"n_calibration={payload['n_calibration']}  n_holdout={payload['n_holdout']}",
        "",
    ]
    for name, result in chosen.items():
        a_text = "none" if result["a"] is None else f"{result['a']:.1f}"
        lines.append(
            f"{name}: a={a_text}  bias={result['bias']:+.3f}  "
            f"cal pred={result['calibration_prediction_mean']:.3f}  "
            f"cal actual={result['calibration_actual_mean']:.3f}"
        )
    lines.extend(["", f"Wrote {json_path}", ""])
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
