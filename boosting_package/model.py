# Frozen NSW demand forecaster for the course's one-day runner

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

if __package__:
    from .features import forecast_features
else:
    from features import forecast_features

# Baseline in MW to which the fitted model's prediction is added.
def demand_offset(features, target):
    if target == "demand":
        return 0.0
    return features[target].to_numpy()


@dataclass
class DemandModel:
    estimator: HistGradientBoostingRegressor
    target: str

    def predict(self, features):
        columns = self.estimator.feature_names_in_
        with threadpool_limits(limits=2, user_api="openmp"):
            prediction = self.estimator.predict(features.loc[:, columns])
        return np.clip(prediction + demand_offset(features, self.target), 0.0, 20000.0)


def load_demand_model(artifact_dir):
    saved = joblib.load(Path(artifact_dir) / "demand_model.joblib")
    return DemandModel(saved["estimator"], saved["target"])


class Forecaster:
    def __init__(self, artifact_dir):
        self.demand = load_demand_model(artifact_dir)

    def predict(self, history, weather_next):
        features = forecast_features(history, weather_next)
        # The standalone demand experiment supplies the kit's yesterday price baseline.
        yesterday_prices = history["nsw_price"].iloc[-48:]
        price_up = (yesterday_prices > yesterday_prices.mean()).to_numpy(dtype=int)
        return pd.DataFrame(
            {"nsw_demand_mw": self.demand.predict(features), "nsw_price_up": price_up},
            index=weather_next.index,
        )
