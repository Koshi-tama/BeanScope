"""入力から結果表示までの煙テスト。"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _text(at: AppTest) -> str:
    return "\n".join((element.value or "") for element in list(at.markdown) + list(at.text) + list(at.caption) + list(at.info))


def test_predict_show_warnings_and_reset() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "BeanScope"
    assert [tab.label for tab in at.tabs] == ["予測", "モデルについて", "データ・注意事項"]

    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].label == "予測品質スコア"
    assert at.metric[0].value == "84.6"
    text = _text(at)
    assert "1.4 点" in text
    assert "信頼区間ではなく" in text
    assert "カッピングを代替しません" in text
    assert "ランダムフォレスト" in text
    assert "1,325.88 m" in text
    assert "Coffee Quality Institute" in text

    at.slider[0].set_value(100)
    at.button[0].click().run()
    assert not at.exception
    assert any("範囲" in warning.value for warning in at.warning)

    at.button[1].click().run()
    assert not at.exception
    assert at.slider[0].value == 1800
    assert not at.metric
