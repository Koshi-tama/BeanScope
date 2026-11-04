"""
予測を定数へ縮める割合 a を、D01 の後半だけで決めて保存する。

学習そのものは models/train.py が行い、このスクリプトは a の比較表を残す。
a の選択に D02 の正解は使わない。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shrinkage import A_GRID, select_shrinkage_factors  # noqa: E402
from train import candidate_models  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "shrinkage"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = select_shrinkage_factors(candidate_models())
    print(
        f"early n={chosen['n_early']}  late n={chosen['n_late']}  "
        f"undated excluded n={chosen['n_undated_excluded']}"
    )
    for name, result in chosen["models"].items():
        print(
            f"{name:24s}  a={result['selected_a']:.1f}  "
            f"late MAE={result['selected_mae']:.3f}  "
            f"center={result['prediction_center']:.3f}"
        )
        for row in result["grid"]:
            mark = " <--" if row["a"] == result["selected_a"] else ""
            print(f"    a={row['a']:.1f}  MAE={row['mae']:.4f}{mark}")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": chosen["cutoff"],
        "a_grid": A_GRID,
        "n_early": chosen["n_early"],
        "n_late": chosen["n_late"],
        "n_undated_excluded": chosen["n_undated_excluded"],
        "rule": (
            "Fit on D01 graded before the cutoff. "
            "Choose a that minimizes MAE on D01 graded on or after the cutoff. "
            "D02 labels are not used."
        ),
        "models": chosen["models"],
    }
    json_path = OUTPUT_DIR / "chosen_a.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "Shrinkage factor a, chosen on D01 from 2016 onward",
        "=" * 60,
        "2016-01-01 より前で学習し、2016年以降の MAE が最小の a を採用する。",
        "D02 の正解は使用していない。",
        f"early n={chosen['n_early']}  late n={chosen['n_late']}  "
        f"undated excluded n={chosen['n_undated_excluded']}",
        "",
    ]
    for name, result in chosen["models"].items():
        lines.append(
            f"{name}: a={result['selected_a']:.1f}  "
            f"late MAE={result['selected_mae']:.3f}  "
            f"center={result['prediction_center']:.3f}"
        )
    lines.extend(["", f"Wrote {json_path}", ""])
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
