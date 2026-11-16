"""Fit and score Bryan's Stage 1 per-slot Lasso across chronological windows."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


# Bryan's pickles use the top-level names model.PerSlotLinearRegressor and
# features.DayTypeTransformer. Preserve those names for his standalone Forecaster.
HERE = Path(__file__).resolve().parent
BRYAN_DIR = HERE if (HERE / "features.py").is_file() else HERE.parent / "bryan_package"
if str(BRYAN_DIR) not in sys.path:
    sys.path.insert(0, str(BRYAN_DIR))
from features import DayTypeTransformer, N_PERIODS, WINDOW_DAYS, build_candidate_features, market_day
from model import DEMAND_MAX, DEMAND_MIN, PerSlotLinearRegressor


@dataclass
class PreparedLasso:
    """One row per target market day, with the same 65 slot features as Stage 1."""

    dates: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    feature_names: list
    selected_indices: list
    transformer: DayTypeTransformer
    config: dict

    def _rows(self, start, end):
        # Dates are inclusive. A historical window may begin during the initial
        # seven lag-only days, which never supply a supervised training row.
        first = self.dates.searchsorted(pd.Timestamp(start), side="left")
        stop = self.dates.searchsorted(pd.Timestamp(end), side="right")
        if first >= stop:
            raise ValueError(f"No eligible forecast days between {start} and {end}.")
        return slice(first, stop)

    def fit(self, start, end):
        """Refit 48 scalers and Lasso models with targets from start through end."""
        rows = self._rows(start, end)
        estimator = PerSlotLinearRegressor(
            model_type=self.config["selected_model_type"],
            alpha=self.config["alpha"],
            l1_ratio=self.config["l1_ratio"],
            selected_feature_names=self.feature_names,
            day_type_transformer=self.transformer,
        )
        # The cached tensor is already in the selected order. Set the full
        # candidate-column indices afterwards for predictions on raw daily inputs.
        with threadpool_limits(limits=1):
            estimator.fit(self.x[rows], self.y[rows])
        estimator.selected_indices = self.selected_indices
        return estimator

    def predict(self, model, start, end):
        """Return day × 48 forecasts, including fast single-day online scoring."""
        # Fold each fitted StandardScaler into its linear model. This evaluates
        # the same prediction as predict_tensor without 48 sklearn calls per day.
        weights = np.stack([
            fit.coef_ / scaler.scale_ for fit, scaler in zip(model.models_, model.scalers_)
        ])
        intercepts = np.array([
            fit.intercept_ - scaler.mean_ @ w
            for fit, scaler, w in zip(model.models_, model.scalers_, weights)
        ])
        predicted = np.einsum("dsf,sf->ds", self.x[self._rows(start, end)], weights) + intercepts
        return np.clip(predicted, DEMAND_MIN, DEMAND_MAX)

    @staticmethod
    def save(model, path):
        """Write demand_model.joblib in the format loaded by Bryan's Forecaster."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)


def prepare_lasso(df, transformer_year=2014, config_path=None):
    """Build causal features once, fitting PCA/K-means only through transformer_year.

    Use 2013 for the initial 2014 validation and 2014 for deployment. Keep the
    transformer fixed across subsequent refits so selected cluster IDs retain
    the meanings they had at deployment.
    """
    config_file = Path(config_path) if config_path else BRYAN_DIR / "artifacts" / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config["selected_model_type"] != "lasso":
        raise ValueError("Expected Bryan's selected Stage 1 Lasso configuration.")
    index = df.index
    if len(index) <= WINDOW_DAYS * N_PERIODS:
        raise ValueError("At least eight complete market days are needed.")
    expected = pd.date_range(index[0].normalize() + pd.Timedelta(minutes=30),
                             periods=len(index), freq="30min")
    if len(index) % N_PERIODS or not index.equals(expected):
        raise ValueError("Expected complete, consecutive end-stamped market days.")

    days = market_day(index[::N_PERIODS])
    all_demand = df["nsw_demand"].to_numpy(dtype=np.float64).reshape(-1, N_PERIODS)
    reference = days.year <= transformer_year
    if not reference.any():
        raise ValueError(f"No data through reference year {transformer_year}.")
    transformer = DayTypeTransformer(n_components=5, n_clusters=4, random_state=42)
    with threadpool_limits(limits=1):
        transformer.fit(all_demand[reference])

    names = config["selected_features"]
    target_dates = days[WINDOW_DAYS:]
    features = np.empty((len(target_dates), N_PERIODS, len(names)), dtype=np.float64)
    # Each year's worth of forecasts uses the same frozen transformer and
    # carries seven history days from the prior block across the boundary.
    indices = None
    for first in range(WINDOW_DAYS, len(days), 365):
        stop = min(first + 365, len(days))
        frame = df.iloc[(first - WINDOW_DAYS) * N_PERIODS : stop * N_PERIODS]
        with threadpool_limits(limits=1):
            block = build_candidate_features(frame, day_type_transformer=transformer)
        if indices is None:
            indices = [block["feature_names"].index(name) for name in names]
        rows = slice(first - WINDOW_DAYS, stop - WINDOW_DAYS)
        features[rows] = block["X_3d"][:, :, indices]

    return PreparedLasso(target_dates, features, all_demand[WINDOW_DAYS:], names,
                         indices, transformer, config)
