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

    def get_coef_matrix(self) -> np.ndarray:
        """Return (48, n_selected_features) matrix of standardised regression coefficients."""
        return np.vstack([m.coef_ for m in self.models_])


class Forecaster:
    """Authoritative submission Forecaster for COMP9417 live evaluation."""

    def __init__(self, artifact_dir: str):
        """Load fitted models and configuration from artifact_dir. No training here."""
        config_path = os.path.join(artifact_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = json.load(fh)
        self.window_days = int(self.config.get("window_days", 7))

        model_path = os.path.join(artifact_dir, "demand_model.joblib")
        self.demand_model: PerSlotLinearRegressor = joblib.load(model_path)

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        """Forecast the 48 half-hours of day D+1.

        Demand (nsw_demand_mw): Predicted using the fitted regularised linear regression
        model with slot-aligned lags, rolling stats, weather deltas, regime interactions,
        interstate variables, and historical day-type cluster features.

        Price direction (nsw_price_up): Same-half-hour-yesterday baseline
        (1 if P_{D,t} > mean(P_D) else 0).
        """
        X_day = build_single_day_features(
            history=history,
            weather_next=weather_next,
            day_type_transformer=self.demand_model.day_type_transformer,
        )
        demand_pred = self.demand_model.predict_single_day(X_day)

        # Same-half-hour-yesterday baseline for price direction
        last_day_prices = history["nsw_price"].iloc[-N_PERIODS:].to_numpy(dtype=np.float64)
        mean_price_day_d = float(np.mean(last_day_prices))
        price_up_pred = (last_day_prices > mean_price_day_d).astype(int)

        return pd.DataFrame(
            {
                "nsw_demand_mw": demand_pred,
                "nsw_price_up": price_up_pred,
            },
            index=weather_next.index,
        )
