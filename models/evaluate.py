"""
保存済み成果物の評価サマリ表示、および Pipeline の簡易推論スモークテスト。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
for _path in (ROOT, ROOT / "input"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from preprocessing import prepare_d02  # noqa: E402
from train import FEATURE_COLS  # noqa: E402


def main() -> None:
    evaluation = json.loads((ARTIFACT_DIR / "evaluation.json").read_text(encoding="utf-8"))
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    pipe = joblib.load(ARTIFACT_DIR / metadata["model_path"])

    print("Selected:", evaluation["selected_model"])
    print("D02 external:", evaluation["external_d02"])
    print("Dummy D02:", evaluation["baseline_dummy_external_d02"])
    print("Acceptance:", evaluation["acceptance"])

    d02, _, _ = prepare_d02()
    sample = d02.head(3)[FEATURE_COLS]
    preds = pipe.predict(sample)
    print("\nSmoke predictions on 3 D02 rows:")
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(
            f"  {row['country']} / {row['process']} -> "
            f"pred={preds[i]:.2f}  actual={d02.loc[idx, 'target_score']:.2f}"
        )


if __name__ == "__main__":
    main()
