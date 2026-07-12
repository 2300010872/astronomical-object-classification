from __future__ import annotations

import numpy as np
import pandas as pd


class QuantileBinner:
    """Quantile-based discretizer for continuous tabular features.

    The binner is fitted on the training set only. Validation/test data must use
    the same fitted bin edges to avoid data leakage.
    """

    def __init__(self, n_bins: int = 192):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_bins = n_bins
        self.edges_: dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame) -> "QuantileBinner":
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.edges_ = {}
        for col in X.columns:
            values = X[col].astype(float).to_numpy()
            edges = np.quantile(values, quantiles)
            # Remove duplicate edges caused by repeated values.
            edges = np.unique(edges)
            if len(edges) <= 2:
                # Degenerate feature: put everything in one bin.
                edges = np.array([np.nanmin(values), np.nanmax(values)])
            self.edges_[col] = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.edges_:
            raise RuntimeError("QuantileBinner is not fitted yet")
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self.edges_:
                raise ValueError(f"Column {col} was not seen during fit")
            edges = self.edges_[col]
            # Internal edges only. digitize returns 0..n_bins-1 approximately.
            internal_edges = edges[1:-1]
            out[col] = np.digitize(X[col].astype(float).to_numpy(), internal_edges, right=False)
        return out.astype(int)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
