"""Model definitions and live evaluation Forecaster interface for my_package.

Defines:
  - PerSlotLinearRegressor: 48 half-hour slot-specific regularised linear regression
    models (supporting Ridge L2, Lasso L1, and ElasticNet L1+L2) with slot-wise
    StandardScaler preprocessing and selected feature subsetting.
  - Forecaster: The authoritative COMP9417 submission class loaded by harness/run_day.py.
"""
import json
import os
import sys
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
FEATURES_DIR = os.path.join(HERE, "artifacts")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from features import (  # noqa: E402
    DayTypeTransformer,
    N_PERIODS,
    build_single_day_features,
)

DEMAND_MIN = 0.0
DEMAND_MAX = 20000.0


class PerSlotLinearRegressor:
    """Slot-aligned regularised linear regression ensemble for 48 half-hourly demand targets.

    Each half-hour slot t in {0, ..., 47} has its own StandardScaler and regularised
    linear estimator (Ridge, Lasso, or ElasticNet), allowing the model to capture
    diurnal variations in thermal sensitivity, rooftop solar suppression, and
    weekday/weekend lag weights.
    """

    def __init__(
        self,
        model_type: str = "lasso",
        alpha: float = 0.50,
        l1_ratio: float = 0.80,
        selected_indices: Optional[List[int]] = None,
        selected_feature_names: Optional[List[str]] = None,
        day_type_transformer: Optional[DayTypeTransformer] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.selected_indices = selected_indices
        self.selected_feature_names = selected_feature_names
        self.day_type_transformer = day_type_transformer
        self.random_state = random_state

        self.scalers_: List[StandardScaler] = []
        self.models_: List[object] = []

    def _make_estimator(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha)
        if self.model_type == "lasso":
            return Lasso(alpha=self.alpha, max_iter=3500, random_state=self.random_state)
        if self.model_type == "elasticnet":
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=3500,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _subset_tensor(self, X_3d: np.ndarray) -> np.ndarray:
        if self.selected_indices is not None:
            return X_3d[:, :, self.selected_indices]
        return X_3d

    def fit(
        self,
        X_3d: np.ndarray,
        Y_2d: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PerSlotLinearRegressor":
        """Fit 48 slot-specific StandardScalers and regularised linear models."""
        X_sub = self._subset_tensor(X_3d)
        self.scalers_ = []
        self.models_ = []

        for t in range(N_PERIODS):
            scaler = StandardScaler()
            xt_scaled = scaler.fit_transform(X_sub[:, t, :], sample_weight=sample_weight)
            estimator = self._make_estimator()
            estimator.fit(xt_scaled, Y_2d[:, t], sample_weight=sample_weight)
            self.scalers_.append(scaler)
            self.models_.append(estimator)
        return self

    def predict_tensor(self, X_3d: np.ndarray) -> np.ndarray:
        """Predict (n_days, 48) demand in MW for a 3D feature tensor."""
        X_sub = self._subset_tensor(X_3d)
        n_days = X_sub.shape[0]
        preds = np.zeros((n_days, N_PERIODS), dtype=np.float64)
        for t in range(N_PERIODS):
            xt_scaled = self.scalers_[t].transform(X_sub[:, t, :])
            preds[:, t] = self.models_[t].predict(xt_scaled)
        return np.clip(preds, DEMAND_MIN, DEMAND_MAX)

    def predict_single_day(self, X_day_2d: np.ndarray) -> np.ndarray:
        """Predict (48,) demand in MW for a single target day's (48, n_features) matrix."""
        if self.selected_indices is not None:
            X_sub = X_day_2d[:, self.selected_indices]
        else:
            X_sub = X_day_2d
        preds = np.zeros(N_PERIODS, dtype=np.float64)
        for t in range(N_PERIODS):
            xt_scaled = self.scalers_[t].transform(X_sub[t : t + 1, :])
            preds[t] = float(self.models_[t].predict(xt_scaled)[0])
        preds = np.nan_to_num(preds, nan=7800.0, posinf=DEMAND_MAX, neginf=DEMAND_MIN)
        return np.clip(preds, DEMAND_MIN, DEMAND_MAX)


def market_day(index):
    return pd.DatetimeIndex(index - pd.Timedelta(minutes=30)).normalize()


def price_features_from_components(t):
    """Shared feature transformations for training and live prediction."""
    X = t[["demand_lag1", "price_lag1", "previous_mean_price",
           "syd_temp", "syd_solar", "syd_rh", "up_lag1", "up_lag7"]].copy()
    for d in range(7):
        X[f"dow_{d}"] = (t["day"].dt.dayofweek == d).astype(int)
    X["sin_period"] = np.sin(2 * np.pi * t["period"] / 48)
    X["cos_period"] = np.cos(2 * np.pi * t["period"] / 48)
    doy = t["day"].dt.dayofyear
    X["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    for prefix, lag in (("demand", "demand_lag1"), ("price", "price_lag1")):
        for cycle in ("period", "year"):
            for trig in ("sin", "cos"):
                X[f"{prefix}_{trig}_{cycle}"] = X[lag] * X[f"{trig}_{cycle}"]
    X["temp_squared"] = X["syd_temp"] ** 2
    X["temp_sin_period"] = X["syd_temp"] * X["sin_period"]
    X["temp_cos_period"] = X["syd_temp"] * X["cos_period"]
    X["price_relative_mean"] = X["price_lag1"] - X["previous_mean_price"]
    X["demand_pct_change"] = ((X["demand_lag1"] - t["demand_lag7"])
                              / t["demand_lag7"].clip(lower=1))
    for trig in ("sin", "cos"):
        X[f"solar_{trig}_period"] = X["syd_solar"] * X[f"{trig}_period"]
    for cycle in ("period", "year"):
        for trig in ("sin", "cos"):
            X[f"up_lag1_{trig}_{cycle}"] = X["up_lag1"] * X[f"{trig}_{cycle}"]
    X["up_lag1_lag7"] = X["up_lag1"] * X["up_lag7"]
    X["up_lag1_price_relative"] = X["up_lag1"] * X["price_relative_mean"]
    for d in range(1, 7):
        X[f"up_lag1_dow_{d}"] = X["up_lag1"] * X[f"dow_{d}"]
    return X


def build_price_features(history, weather_next):
    h = history.sort_index()
    w = weather_next.sort_index()
    if len(w) != 48 or len(h) < 48 * 7:
        raise ValueError("Need 48 next-day intervals and at least seven historical days")
    if not h.index.is_unique or not w.index.is_unique:
        raise ValueError("Duplicate timestamps")
    hd, wd = market_day(h.index), market_day(w.index)
    if len(pd.unique(wd)) != 1 or wd[0] != hd[-1] + pd.Timedelta(days=1):
        raise ValueError("Next-day weather is not aligned to history")
    prev = h.iloc[-48:]
    week = h.iloc[-48 * 7:-48 * 6]
    previous_mean = float(prev["nsw_price"].mean())
    t = pd.DataFrame(index=w.index)
    t["day"] = wd
    shifted = w.index - pd.Timedelta(minutes=30)
    t["period"] = shifted.hour * 2 + shifted.minute // 30
    t["demand_lag1"] = prev["nsw_demand"].to_numpy()
    t["demand_lag7"] = week["nsw_demand"].to_numpy()
    t["price_lag1"] = prev["nsw_price"].to_numpy()
    t["previous_mean_price"] = previous_mean
    t["up_lag1"] = (prev["nsw_price"].to_numpy() > previous_mean).astype(int)
    t["up_lag7"] = (week["nsw_price"].to_numpy() > previous_mean).astype(int)
    for col in ("syd_temp", "syd_solar", "syd_rh"):
        t[col] = w[col].to_numpy()
    return price_features_from_components(t)


class Forecaster:
    """Combined COMP9417 demand and price-direction submission interface."""

    def __init__(self, artifact_dir: str):
        config_path = os.path.join(artifact_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = json.load(fh)
        self.demand_model: PerSlotLinearRegressor = joblib.load(
            os.path.join(artifact_dir, "demand_model.joblib")
        )
        self.price_model = joblib.load(
            os.path.join(artifact_dir, "price_model.joblib")
        )

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        demand_features = build_single_day_features(
            history=history,
            weather_next=weather_next,
            day_type_transformer=self.demand_model.day_type_transformer,
        )
        demand = self.demand_model.predict_single_day(demand_features)

        price_features = build_price_features(history, weather_next)
        expected = list(self.price_model.feature_names_in_)
        probabilities = self.price_model.predict_proba(price_features[expected])[:, 1]
        price = (probabilities > 0.5).astype(int)

        return pd.DataFrame(
            {"nsw_demand_mw": demand, "nsw_price_up": price},
            index=weather_next.index,
        )
