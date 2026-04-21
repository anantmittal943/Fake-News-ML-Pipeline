from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SentinelToNaN(BaseEstimator, TransformerMixin):
    """Replace a fixed sentinel value in a column with NaN."""

    def __init__(self, column: str, sentinel: float = 0.0):
        self.column = column
        self.sentinel = sentinel

    def fit(self, X: pd.DataFrame, y=None) -> "SentinelToNaN":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.column] = out[self.column].replace(self.sentinel, np.nan)
        return out
