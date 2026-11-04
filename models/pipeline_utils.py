"""学習 Pipeline 用の共通変換器。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """学習データ内の出現回数 < min_count のカテゴリを Other へまとめる。"""

    def __init__(self, min_count: int = 10, other_label: str = "Other"):
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=object)
        self.n_features_in_ = arr.shape[1]
        self.keep_levels_: list[set[str]] = []
        for i in range(arr.shape[1]):
            col = pd.Series(arr[:, i]).astype("string").fillna("Missing")
            counts = col.value_counts()
            self.keep_levels_.append(
                set(counts[counts >= self.min_count].index.tolist())
            )
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object).copy()
        for i in range(arr.shape[1]):
            col = pd.Series(arr[:, i]).astype("string").fillna("Missing")
            arr[:, i] = np.where(
                col.isin(self.keep_levels_[i]),
                col.to_numpy(),
                self.other_label,
            )
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(
            [f"x{i}" for i in range(self.n_features_in_)], dtype=object
        )
