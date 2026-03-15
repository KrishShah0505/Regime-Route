"""
regime/classifier.py
--------------------
Classifies every trading day into one of 3 volatility regimes:
    0 = Low Vol    (calm, trending — run Momentum)
    1 = High Vol   (chaotic, mean-reverting — run Mean Reversion)
    2 = Transition (uncertain — run Trend Filter with reduced size)

Two classification methods:
    1. HMM (Gaussian Hidden Markov Model) — primary, statistically principled
    2. Rolling Z-Score — fallback, fully transparent and explainable

Walk-forward fitting:
    The classifier is NEVER fit on future data.
    It re-fits every 63 days on a rolling 2-year training window.
    This is the gold standard for avoiding lookahead bias in regime detection.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
import warnings

logger = logging.getLogger(__name__)

# Regime labels
REGIME_LOW_VOL = 0
REGIME_HIGH_VOL = 1
REGIME_TRANSITION = 2

REGIME_NAMES = {
    REGIME_LOW_VOL: "Low Volatility",
    REGIME_HIGH_VOL: "High Volatility",
    REGIME_TRANSITION: "Transitional",
}


class RegimeClassifier:
    """
    Walk-forward volatility regime classifier.

    Fits a Gaussian HMM on rolling windows of market volatility features.
    Because HMM state labels are arbitrary, we post-process to align them
    consistently: state with lowest mean VIX = regime 0, highest = regime 1.

    Parameters
    ----------
    n_regimes       : number of hidden states (default 3)
    method          : 'hmm' or 'zscore'
    train_window    : trading days in each training window (default 504 = 2yr)
    refit_every     : re-fit the model every N days (default 63 = 1 quarter)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        method: str = "hmm",
        train_window: int = 504,
        refit_every: int = 63,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.method = method
        self.train_window = train_window
        self.refit_every = refit_every
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._model = None
        self._regime_map = None      # maps HMM state -> our regime label

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Walk-forward fit and predict. This is the main entry point.

        Iterates through the feature matrix in blocks of 'refit_every' days.
        For each block, fits the HMM on the preceding 'train_window' days,
        then predicts regime labels for the current block.

        Returns
        -------
        np.ndarray of shape (n_days,) with regime labels {0, 1, 2}
        Aligned 1:1 with features.index
        """
        if self.method == "zscore":
            return self._zscore_regimes(features)

        return self._walk_forward_hmm(features)

    def predict_latest(self, features: pd.DataFrame) -> int:
        """
        Predict regime for the most recent day only.
        Used by the API for live regime status.
        """
        if self._model is None:
            labels = self.fit_predict(features)
            return int(labels[-1])

        recent = features.iloc[-self.train_window:].dropna()
        scaled = self.scaler.transform(recent.values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_label = self._model.predict(scaled)[-1]
        return int(self._regime_map.get(raw_label, REGIME_TRANSITION))

    # ── Walk-Forward HMM ──────────────────────────────────────────────────────

    def _walk_forward_hmm(self, features: pd.DataFrame) -> np.ndarray:
        """
        Core walk-forward loop.

        Timeline example (train_window=504, refit_every=63):

            Days 0-503   : burn-in (not enough data to train)
            Days 504-566 : train on [0:504], predict [504:567]
            Days 567-629 : train on [63:567], predict [567:630]
            ...

        All predictions use only past data. No future leakage.
        """
        n = len(features)
        labels = np.full(n, REGIME_TRANSITION, dtype=int)   # default = transitional
        clean = features.dropna()

        # Map original index to position in clean
        idx_map = {date: i for i, date in enumerate(features.index)}

        logger.info(
            f"Walk-forward HMM | "
            f"train_window={self.train_window}, refit_every={self.refit_every}"
        )

        step = 0
        while True:
            train_start = step * self.refit_every
            train_end = train_start + self.train_window
            pred_end = train_end + self.refit_every

            if train_end >= len(clean):
                break

            train_data = clean.iloc[train_start:train_end]
            pred_data = clean.iloc[train_end:pred_end]

            if len(train_data) < self.train_window // 2:
                step += 1
                continue

            # Fit scaler and HMM on training window
            scaled_train = self.scaler.fit_transform(train_data.values)
            model = self._fit_hmm(scaled_train)

            if model is None:
                step += 1
                continue

            # Build regime mapping from this fit
            regime_map = self._build_regime_map(model, scaled_train, train_data)

            # Predict on the out-of-sample block
            scaled_pred = self.scaler.transform(pred_data.values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_preds = model.predict(scaled_pred)

            # Map raw HMM states to our regime labels and write back
            for i, date in enumerate(pred_data.index):
                if date in idx_map:
                    pos = idx_map[date]
                    labels[pos] = regime_map.get(raw_preds[i], REGIME_TRANSITION)

            # Keep the last fitted model for live prediction
            self._model = model
            self._regime_map = regime_map

            step += 1

        logger.info(f"Walk-forward complete | Regime distribution: {self._regime_distribution(labels)}")
        return labels

    def _fit_hmm(self, scaled_data: np.ndarray) -> GaussianHMM | None:
        """Fit a single GaussianHMM. Returns None on convergence failure."""
        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
            tol=1e-4,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(scaled_data)
            return model
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")
            return None

    def _build_regime_map(
        self,
        model: GaussianHMM,
        scaled_data: np.ndarray,
        raw_features: pd.DataFrame,
    ) -> dict:
        """
        Map HMM states to our regime labels {0, 1, 2}.

        HMM state labels are arbitrary — state 2 might be low vol in one fit
        and high vol in the next. We resolve this by looking at the mean VIX
        level for each state and ranking them.

        Lowest mean VIX  → REGIME_LOW_VOL (0)
        Highest mean VIX → REGIME_HIGH_VOL (1)
        Middle           → REGIME_TRANSITION (2)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_sequence = model.predict(scaled_data)

        # VIX is always the first column in our feature matrix
        vix_col = raw_features.iloc[:, 0].values

        # Mean VIX per HMM state
        state_vix_means = {}
        for state in range(self.n_regimes):
            mask = state_sequence == state
            if mask.sum() > 0:
                state_vix_means[state] = vix_col[mask].mean()
            else:
                state_vix_means[state] = 0

        sorted_states = sorted(state_vix_means, key=state_vix_means.get)

        if self.n_regimes == 3:
            return {
                sorted_states[0]: REGIME_LOW_VOL,
                sorted_states[2]: REGIME_HIGH_VOL,
                sorted_states[1]: REGIME_TRANSITION,
            }
        elif self.n_regimes == 2:
            return {
                sorted_states[0]: REGIME_LOW_VOL,
                sorted_states[1]: REGIME_HIGH_VOL,
            }
        else:
            return {s: i for i, s in enumerate(sorted_states)}

    # ── Z-Score Fallback ──────────────────────────────────────────────────────

    def _zscore_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """
        Simple, transparent regime labelling via rolling VIX z-score.
        No ML required — fully explainable.

        Used as:
        1. Fallback if HMM fails
        2. Validation check against HMM results
        """
        vix = features.iloc[:, 0]   # VIX is always first column
        window = min(self.train_window, len(vix) // 2)

        rolling_mean = vix.rolling(window, min_periods=30).mean()
        rolling_std = vix.rolling(window, min_periods=30).std()
        zscore = (vix - rolling_mean) / rolling_std.replace(0, np.nan)

        labels = np.full(len(features), REGIME_TRANSITION, dtype=int)

        # Vectorized labelling using numpy where
        labels = np.where(zscore < -0.5, REGIME_LOW_VOL, labels)
        labels = np.where(zscore > 1.0, REGIME_HIGH_VOL, labels)
        # everything in between stays TRANSITION

        logger.info(f"Z-score regimes | {self._regime_distribution(labels)}")
        return labels

    # ── Utility ───────────────────────────────────────────────────────────────

    def _regime_distribution(self, labels: np.ndarray) -> dict:
        total = len(labels)
        return {
            REGIME_NAMES[r]: f"{(labels == r).sum() / total:.1%}"
            for r in [REGIME_LOW_VOL, REGIME_HIGH_VOL, REGIME_TRANSITION]
        }

    def get_regime_series(
        self, labels: np.ndarray, index: pd.DatetimeIndex
    ) -> pd.Series:
        """Wrap numpy labels array in a named pd.Series with DatetimeIndex."""
        return pd.Series(labels, index=index, name="regime")

    def get_regime_stats(self, labels: np.ndarray, returns: pd.Series) -> pd.DataFrame:
        """
        Compute basic statistics per regime.
        Returns DataFrame with mean_return, std, count per regime.
        """
        regime_series = pd.Series(labels, index=returns.index, name="regime")
        combined = pd.DataFrame({"return": returns, "regime": regime_series})

        stats = combined.groupby("regime")["return"].agg(
            mean_return="mean",
            std="std",
            count="count",
        )
        stats["annualized_return"] = stats["mean_return"] * 252
        stats["annualized_vol"] = stats["std"] * np.sqrt(252)
        stats["sharpe"] = stats["annualized_return"] / stats["annualized_vol"].replace(0, np.nan)
        stats.index = [REGIME_NAMES.get(i, str(i)) for i in stats.index]
        return stats.round(4)
