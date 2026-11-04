"""BeanScope: アラビカ種の品質スコアを画面から予測する。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    FEATURE_LABELS,
    SAMPLE_VALUES,
    Artifacts,
    ModelLoadError,
    ValidationError,
    categories_for,
    load_artifacts,
    numeric_range,
    predict_score,
    validate_input,
)

MODEL_NAMES = {
    "random_forest": "ランダムフォレスト",
    "ridge": "リッジ回帰",
    "hist_gradient_boosting": "ヒストグラム勾配ブースティング",
    "dummy_median": "中央値（比較用）",
}

st.set_page_config(page_title="BeanScope", page_icon="☕", layout="centered")


@st.cache_resource
def cached_artifacts() -> Artifacts:
    return load_artifacts()


def main() -> None:
    try:
        artifacts = cached_artifacts()
    except ModelLoadError as exc:
        st.title("BeanScope")
        st.error(str(exc) + " 時間をおいて開き直してください。")
        st.stop()

    st.title("BeanScope")
    st.write(
        "生産国、標高、品種、精製方法などの生豆条件から、"
        "アラビカ種のカッピング総合点を予測します。"
    )
    st.caption("アラビカ種専用  ·  参考予測（品質認証や取引判断には使えません）")

    home, about, data = st.tabs(["予測", "モデルについて", "データ・注意事項"])
    with home:
        render_home(artifacts)
    with about:
        render_about(artifacts)
    with data:
        render_data(artifacts)


def render_home(artifacts: Artifacts) -> None:
    st.subheader("条件を入力")
    if "form_nonce" not in st.session_state:
        st.session_state.form_nonce = 0

    nonce = st.session_state.form_nonce
    with st.form(f"predict_form_{nonce}"):
        values = _input_widgets(artifacts, nonce)
        submitted = st.form_submit_button("品質スコアを予測", type="primary")

    if st.button("入力を初期値に戻す"):
        st.session_state.form_nonce += 1
        st.session_state.pop("last_prediction", None)
        st.rerun()

    if submitted:
        try:
            checked = validate_input(artifacts, values)
            score = predict_score(artifacts, checked)
        except ValidationError as exc:
            st.session_state.pop("last_prediction", None)
            for message in exc.messages:
                st.error(message)
        except ModelLoadError as exc:
            st.session_state.pop("last_prediction", None)
            st.error(str(exc))
        else:
            st.session_state.last_prediction = {
                "score": score,
                "summary": checked.summary,
                "warnings": checked.warnings,
            }

    last = st.session_state.get("last_prediction")
    if last is not None:
        _render_result(artifacts, last)


def _input_widgets(artifacts: Artifacts, nonce: int) -> dict:
    countries = categories_for(artifacts, "country")
    varieties = categories_for(artifacts, "variety")
    processes = categories_for(artifacts, "process")
    colors = categories_for(artifacts, "color")
    alt_low, alt_high = numeric_range(artifacts, "altitude_m")
    moist_low, moist_high = numeric_range(artifacts, "moisture_pct")
    c1_low, c1_high = numeric_range(artifacts, "category_1_defects")
    c2_low, c2_high = numeric_range(artifacts, "category_2_defects")

    country = st.selectbox(
        "生産国",
        countries,
        index=_index(countries, SAMPLE_VALUES["country"]),
        help="学習データに含まれる生産国だけを選べます。",
        key=f"country_{nonce}",
    )
    altitude = st.slider(
        "栽培標高（m）",
        min_value=0,
        max_value=3000,
        value=int(SAMPLE_VALUES["altitude_m"]),
        step=10,
        help="栽培地の標高です。0〜3,000 m のあいだで動かせます。学習範囲の外でも予測でき、そのときは警告を出します。",
        key=f"altitude_{nonce}",
    )
    st.caption(f"学習データの範囲: {_range_text(alt_low, alt_high)} m")

    variety = st.selectbox(
        "品種",
        varieties,
        index=_index(varieties, SAMPLE_VALUES["variety"]),
        help="表記ゆれをそろえた品種名です。",
        key=f"variety_{nonce}",
    )
    process = st.selectbox(
        "精製方法",
        processes,
        index=_index(processes, SAMPLE_VALUES["process"]),
        help="水洗（Washed）、ナチュラル、ハニーなどに統合した区分です。",
        key=f"process_{nonce}",
    )
    moisture = st.slider(
        "水分率（%）",
        min_value=0.0,
        max_value=40.0,
        value=float(SAMPLE_VALUES["moisture_pct"]),
        step=0.1,
        format="%.1f",
        help="生豆に含まれる水分の割合です。0 は欠損ではなく 0% として扱います。0〜40% のあいだで動かせます。学習範囲の外では警告を出します。",
        key=f"moisture_{nonce}",
    )
    st.caption(f"学習データの範囲: {_range_text(moist_low, moist_high)} %")

    defects_1 = st.number_input(
        "Category 1 欠点数",
        min_value=0,
        value=int(SAMPLE_VALUES["category_1_defects"]),
        step=1,
        help="味を大きく損なう欠点豆の個数です。",
        key=f"c1_{nonce}",
    )
    st.caption(f"学習データの範囲: {_range_text(c1_low, c1_high)}")
    defects_2 = st.number_input(
        "Category 2 欠点数",
        min_value=0,
        value=int(SAMPLE_VALUES["category_2_defects"]),
        step=1,
        help="Category 1 以外の欠点豆の個数です。",
        key=f"c2_{nonce}",
    )
    st.caption(f"学習データの範囲: {_range_text(c2_low, c2_high)}")

    color_options = ["未指定", *colors]
    color = st.selectbox(
        "生豆の色（任意）",
        color_options,
        index=_index(color_options, SAMPLE_VALUES["color"]),
        help="未指定のときは、学習データでもっとも多かった色で補います。",
        key=f"color_{nonce}",
    )
    return {
        "country": country,
        "altitude_m": altitude,
        "variety": variety,
        "process": process,
        "moisture_pct": moisture,
        "category_1_defects": defects_1,
        "category_2_defects": defects_2,
        "color": None if color == "未指定" else color,
    }


def _render_result(artifacts: Artifacts, last: dict) -> None:
    score = float(last["score"])
    holdout = artifacts.evaluation["external_d02"]
    dummy = artifacts.evaluation["baseline_dummy_external_d02"]
    st.subheader("予測結果")
    st.metric("予測品質スコア", f"{score:.1f}")
    st.write(
        f"平均的な絶対誤差は **{holdout['mae']:.1f} 点** です。"
        "これは信頼区間ではなく、定数を決めた 30 件を除く 177 件で、"
        "予測と実際の点数の差を平均した値です。"
    )
    st.caption(
        f"同じ 177 件で、常に中央値を返す比較モデルの平均絶対誤差は {dummy['mae']:.1f} 点です。"
    )
    if score < 0 or score > 100:
        st.warning("予測値が 0〜100 点の外です。点数の目安としてだけ見てください。")

    for message in last["warnings"]:
        st.warning(message)

    st.markdown("**入力した条件**")
    for label, value in last["summary"]:
        st.write(f"- {label}: {value}")

    st.info(
        "この予測は実物のカッピングを代替しません。"
        "品質認証、取引価格、焙煎後の味の判断には使わないでください。"
    )


def render_about(artifacts: Artifacts) -> None:
    meta = artifacts.metadata
    evaluation = artifacts.evaluation
    name = meta["selected_model"]
    reg = artifacts.model.named_steps["model"]
    forest = reg.estimator_
    holdout = evaluation["external_d02"]
    dummy = evaluation["baseline_dummy_external_d02"]
    calibration = meta["level_calibration"]
    cv = evaluation["cv_d01_group_kfold"][name]

    st.subheader("採用したモデル")
    st.write(
        f"**{MODEL_NAMES.get(name, name)}** を、D01 のアラビカ種 "
        f"**{meta['n_production_train']:,} 件**（評価日はおおよそ 2010年4月〜2018年1月）で学習しています。"
    )
    if hasattr(forest, "n_estimators"):
        st.write(
            f"木の数 {forest.n_estimators}、深さの上限 {forest.max_depth}、"
            f"葉の最小サンプル数 {forest.min_samples_leaf}。"
            f"モデル版 {meta['version']}（学習日 {str(meta['trained_at_utc'])[:10]}）。"
        )
    else:
        st.write(f"モデル版 {meta['version']}（学習日 {str(meta['trained_at_utc'])[:10]}）。")
    opening = "まずモデルが点数を出します。" if name == "random_forest" else "まずモデルが点数を出します。"
    st.write(
        f"予測は次の順です。{opening}"
        f"次に、学習時の予測平均 {_one(reg.center_)} 点との差の "
        f"{reg.a_ * 100:.0f}% だけを残すように縮めます。"
        f"最後に、新しい時期の水準として **{reg.bias_:+.2f} 点** を足します。"
        "この定数は、評価日が早い D02 の "
        f"{calibration['n_calibration']} 件"
        f"（{calibration['calibration_start']} 〜 {calibration['calibration_end']}）だけで決めています。"
        "残りの評価用データでは再学習していません。"
    )

    st.subheader("評価指標")
    st.caption(
        "アプリに表示する誤差は、定数を足したあとの評価用 "
        f"{calibration['n_holdout']} 件です。MAE は平均的な絶対誤差、"
        "RMSE は大きいずれをより重く見た誤差、R² は平均だけで予測した場合に比べた説明の割合です。"
    )
    rows = pd.DataFrame(
        [
            {
                "モデル": MODEL_NAMES.get(name, name),
                "MAE": holdout["mae"],
                "RMSE": holdout["rmse"],
                "R²": holdout["r2"],
            },
            {
                "モデル": "中央値（比較用）",
                "MAE": dummy["mae"],
                "RMSE": dummy["rmse"],
                "R²": dummy["r2"],
            },
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        f"D01 のグループ交差検証では、同じモデルの MAE は "
        f"{cv['mae_mean']:.3f} ± {cv['mae_std']:.3f} でした。"
    )

    st.subheader("特徴量の重要度")
    st.caption(
        "評価用データで、その列だけをシャッフルしたときに MAE がどれだけ増えたかです。"
        "品質への因果効果ではありません。左へ伸びる列は、崩しても誤差が増えなかったものです。"
    )
    chart = pd.DataFrame(
        {
            "特徴量": [FEATURE_LABELS.get(row["feature"], row["feature"]) for row in artifacts.importance],
            "MAEの増加": [row["importance_mean"] for row in artifacts.importance],
        }
    ).set_index("特徴量")
    st.bar_chart(chart, horizontal=True)

    used = "、".join(FEATURE_LABELS[column] for column in meta["feature_columns"])
    st.write(f"使った項目: {used}。")
    st.write(
        "香りや酸味など、総合点を構成する官能評価の点数は使っていません。"
        "カッピング前には分からないためです。"
    )


def render_data(artifacts: Artifacts) -> None:
    meta = artifacts.metadata
    excluded = meta.get("d01_exclusions", {})
    st.subheader("データの出典")
    st.markdown(
        "- 元の評価: Coffee Quality Institute\n"
        "- D01（学習）: [TidyTuesday 2020-07-07 Coffee Ratings]"
        "(https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-07-07)\n"
        "- D02（水準の定数と評価）: "
        "[Coffee Quality Data (CQI May-2023)]"
        "(https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)"
        " / [配布リポジトリ](https://github.com/fatih-boyar/coffee-quality-data-CQI)"
    )
    st.write(
        "配布データのライセンスは MIT・CCO と判断しています（確認日 2026-10-20）。"
        "学習とこのアプリでの利用ができる、という前提です。"
    )

    fills = artifacts.imputation_values
    st.subheader("加工の内容")
    st.markdown(
        "- 対象はアラビカ種だけです。ロブスタは学習から除いています"
        f"（D01 で {int(excluded.get('robusta', 0))} 件）。\n"
        "- 総合点が 0 の行と、生産国がない行も学習から除いています。\n"
        "- 標高はメートルにそろえ、200〜2,500 m の外は欠損にしたうえで行は残しています。\n"
        "- 水分率 0 は 0% のままです。欠損にはしていません。\n"
        "- 欠損の補完は学習データだけで決め、予測時も同じ値を使います。"
        f"標高の欠損は中央値の **{float(fills['altitude_m']):,.2f} m**、"
        f"品種はもっとも多い **{fills['variety']}**、"
        f"精製方法はもっとも多い **{fills['process']}**、"
        "生豆の色はもっとも多い "
        f"**{fills['color']}** です。画面で生豆の色を未指定にしたときも、この色で補います。\n"
        "- 出現が 10 件未満の区分は Other にまとめ、カテゴリは数値化してから学習しています。\n"
        f"- 公開モデルの学習件数は {meta['n_production_train']:,} 件です。"
    )

    st.subheader("予測できること、できないこと")
    st.markdown(
        "- 生豆の生産条件から、カッピング総合点の参考値を出します。\n"
        "- ロブスタ、焙煎後の味、抽出液の味は対象外です。\n"
        "- 品質認証、等級、取引価格は判定しません。\n"
        "- 入力した条件はこのアプリに保存しません。"
    )
    st.info(
        "業務で使う場合も、実物のカッピングと併用してください。"
        "BeanScope だけで品質や取引を決めないでください。"
    )


def _index(options: list[str], value: str) -> int:
    if value in options:
        return options.index(value)
    return 0


def _range_text(low: float, high: float) -> str:
    return f"{_trim(low)}〜{_trim(high)}"


def _trim(number: float) -> str:
    if float(number).is_integer():
        return f"{int(number):,}"
    return f"{number:.1f}"


def _one(number: float) -> str:
    return f"{number:.1f}"


main()
