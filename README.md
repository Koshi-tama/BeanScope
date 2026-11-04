# BeanScope

アラビカ種の生豆条件から、カッピング総合点の参考値を予測する Streamlit アプリです。品質認証や取引判断には使いません。

## 起動

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで開いたら、初期値のまま「品質スコアを予測」を押すと結果が出ます。入力内容は保存しません。

テストを実行する場合は、開発用の依存関係も入れます。

```bash
pip install -r requirements-dev.txt
pytest
```

## データ出典

| データ | 役割 | 出典 |
|--------|------|------|
| D01 | 学習（整形後 1,309 件） | [TidyTuesday 2020-07-07 Coffee Ratings](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-07-07) |
| D02 | 水準の定数 30 件と採点 177 件 | [Coffee Quality Data (CQI May-2023)](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi) / [配布リポジトリ](https://github.com/fatih-boyar/coffee-quality-data-CQI) |

元の評価は Coffee Quality Institute のものです。配布データのライセンスは MIT と判断しています（確認日 2026-10-20。詳細は `doc/data_ledger.md`）。

## モデル概要

公開モデルは `models/artifacts/beanscope_pipeline.joblib` です。

- アルゴリズムはランダムフォレストです。D01 の 1,309 件だけで学習しています。
- 予測は、学習時の予測平均との差の 40% だけを残すように縮め、そのあと **+1.15 点**を足します。
- この定数は、評価日が早い D02 の 30 件（2022-05-11 〜 2022-07-11）だけで決めています。残り 177 件では再学習していません。
- アプリに出す平均的な絶対誤差は、その 177 件の MAE **1.37 点**です。同じ行で中央値だけを返す比較モデルは **1.78 点**です。
- 欠損は学習時の値で補います。標高は中央値 1,325.88 m、品種は Caturra、精製方法は Washed、生豆の色は Green です。

学習スクリプトは `models/train.py`、縮約と定数を決めるスクリプトは `models/choose_shrinkage.py` と `models/choose_calibration.py` です。
