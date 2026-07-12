from __future__ import annotations

import numpy as np
import pandas as pd


class WeightedHistogramNB:
    """Histogram Naive Bayes classifier with optional feature weights.

    Input features should already be discretized into integer bins.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        uniform_prior: bool = True,
        feature_weights: dict[str, float] | None = None,
    ):
        self.alpha = alpha
        self.uniform_prior = uniform_prior
        self.feature_weights = feature_weights or {}
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[str, np.ndarray]] = {}
        self.n_bins_: dict[str, int] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeightedHistogramNB":
        self.feature_names_ = list(X.columns)
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        n_classes = len(self.classes_)
        class_counts = y.value_counts().to_dict()

        if self.uniform_prior:
            self.class_log_prior_ = {c: -np.log(n_classes) for c in self.classes_}
        else:
            total = len(y)
            self.class_log_prior_ = {
                c: np.log(class_counts.get(c, 0) / total) for c in self.classes_
            }

        self.feature_log_prob_ = {c: {} for c in self.classes_}
        self.n_bins_ = {col: int(X[col].max()) + 1 for col in self.feature_names_}

        for c in self.classes_:
            mask = y == c
            X_c = X.loc[mask]
            for col in self.feature_names_:
                n_bins = self.n_bins_[col]
                counts = np.bincount(X_c[col].to_numpy(), minlength=n_bins).astype(float)
                smoothed = counts + self.alpha
                probs = smoothed / smoothed.sum()
                self.feature_log_prob_[c][col] = np.log(probs)
        return self

    def _score_one_class(self, X: pd.DataFrame, c: str) -> np.ndarray:
        scores = np.full(len(X), self.class_log_prior_[c], dtype=float)
        for col in self.feature_names_:
            weight = float(self.feature_weights.get(col, 1.0))
            bins = X[col].to_numpy()
            log_prob = self.feature_log_prob_[c][col]
            # Safety for unseen extreme bins after transform.
            bins = np.clip(bins, 0, len(log_prob) - 1)
            scores += weight * log_prob[bins]
        return scores

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet")
        return np.vstack([self._score_one_class(X, c) for c in self.classes_]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.decision_function(X)
        best_idx = scores.argmax(axis=1)
        return self.classes_[best_idx]
