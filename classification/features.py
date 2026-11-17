"""Shared feature builder for the day-ahead task (price direction and demand).

One row per half-hour of the target day D+1. Every feature uses only market data up to the
end of day D, plus the weather of D+1 and calendar information (spec, "Prediction Task").

Two entry points, one code path, so experiments and the package compute identical features:

    build_table(df)                          every day in the CSV, for experiments
    features_for_day(history, weather_next)  the 48 rows of one forecast, for model.py

Both return every defined feature; each model script selects its own list with t[FEATURES].
Self-contained (numpy, pandas, holidays only) so it can be copied into a submission package.

    import features as F
    t = F.build_table(F.load("data/nem_weather_halfhourly_2010_2026.csv"))
    X, y = t[FEATURES], t["y_up"]

Self-test (checks both entry points agree):
    python features.py data/nem_weather_halfhourly_2010_2026.csv
"""
import holidays
import numpy as np
import pandas as pd

# Feature set shared by the classifiers, so they are compared on identical inputs.
SHARED_FEATURES = [
    "demand_lag1", "price_lag1", "previous_mean_price",
    "syd_temp", "syd_solar", "syd_rh", "up_lag1", "up_lag7",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "sin_period", "cos_period", "sin_year", "cos_year",
    "demand_sin_period", "demand_cos_period", "price_sin_period", "price_cos_period",
    "demand_sin_year", "demand_cos_year", "price_sin_year", "price_cos_year",
    "temp_squared", "temp_sin_period", "temp_cos_period",
    "price_relative_mean", "demand_pct_change",
    "solar_sin_period", "solar_cos_period",
    "up_lag1_sin_period", "up_lag1_cos_period", "up_lag1_sin_year", "up_lag1_cos_year",
    "up_lag1_lag7", "up_lag1_price_relative",
    "up_lag1_dow_1", "up_lag1_dow_2", "up_lag1_dow_3",
    "up_lag1_dow_4", "up_lag1_dow_5", "up_lag1_dow_6",
    "is_holiday",
]

# The shared set split into groups by meaning, for ablations and importance analysis.
FEATURE_GROUPS = {
    "price direction (day D)": ["up_lag1", "up_lag7", "up_lag1_lag7"],
    "price level (day D)": ["price_lag1", "previous_mean_price", "price_relative_mean"],
    "demand (day D)": ["demand_lag1", "demand_pct_change"],
    "time of day": ["sin_period", "cos_period"],
    "calendar": [f"dow_{d}" for d in range(7)] + ["is_holiday", "sin_year", "cos_year"],
    "weather (D+1)": ["syd_temp", "syd_solar", "syd_rh", "temp_squared"],
}
FEATURE_GROUPS["interactions"] = [f for f in SHARED_FEATURES
                                  if not any(f in g for g in FEATURE_GROUPS.values())]

TARGETS = ["y_up", "y_demand"]
META = ["day", "year", "period"]

N = 48                      # half-hours per day
LAG1, LAG7 = N, 7 * N       # same half-hour on day D, and on day D-6
WEATHER_COLS = [f"{c}_{v}" for c in ("syd", "mel", "bne") for v in ("temp", "rh", "cloud", "solar")]


# ---- helpers -------------------------------------------------------------------------------

def load(path):
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").sort_index()


def market_day(index):
    """Day each interval belongs to (timestamps mark the interval END); same as dayahead.py."""
    return (index - pd.Timedelta("30min")).normalize()


def stage1_masks(t):
    """Spec Stage 1: train on 2010-2013, select on 2014."""
    return (t["year"] <= 2013).to_numpy(), (t["year"] == 2014).to_numpy()


def simulation_mask(t):
    """Spec Stage 2: January 2015 to October 2026."""
    return (t["year"] >= 2015).to_numpy()


def _nsw_holidays(days):
    years = range(days.min().year, days.max().year + 1)
    hol = holidays.country_holidays("AU", subdiv="NSW", years=years)
    return days.isin(pd.to_datetime(list(hol.keys())))


# ---- every feature is defined here ---------------------------------------------------------

def _build(df):
    """Features and targets for every row of df. Requires whole, contiguous days.

    Row r belongs to target day D+1; shift(LAG1) reads the same half-hour of day D,
    shift(LAG7) the same half-hour of day D-6."""
    out = pd.DataFrame(index=df.index)
    start = df.index - pd.Timedelta("30min")
    day = market_day(df.index)
    out["day"] = day
    out["year"] = day.year
    out["period"] = start.hour * 2 + start.minute // 30          # 0..47

    price, demand = df["nsw_price"], df["nsw_demand"]
    by_day = price.groupby(day)
    day_mean = by_day.transform("mean")
    day_std = by_day.transform("std", ddof=0)
    prev_mean = day_mean.shift(LAG1)                              # mean price of day D
    prev_std = day_std.shift(LAG1).clip(lower=1.0)                # $/MWh; avoids /0 on flat days

    # targets (NaN on the forecast day, where prices are unknown)
    out["y_demand"] = demand
    out["y_up"] = (price > prev_mean).astype(float).where(price.notna() & prev_mean.notna())

    # demand (MW)
    out["demand_lag1"] = demand.shift(LAG1)                       # same half-hour, day D
    out["demand_lag7"] = demand.shift(LAG7)                       # same half-hour, day D-6
    out["demand_change"] = out["demand_lag1"] - out["demand_lag7"]

    # price levels ($/MWh) - drift with the market
    out["price_lag1"] = price.shift(LAG1)
    out["price_lag7"] = price.shift(LAG7)
    out["price_change"] = out["price_lag1"] - out["price_lag7"]
    out["previous_mean_price"] = prev_mean

    # price relative to day D's mean - same frame as the target, scale-free
    out["up_lag1"] = (out["price_lag1"] > prev_mean).astype(float)   # = yesterday baseline
    out["up_lag7"] = (out["price_lag7"] > prev_mean).astype(float)   # = last-week baseline
    out["price_rel_lag1"] = (out["price_lag1"] - prev_mean) / prev_std
    out["price_rel_lag7"] = (out["price_lag7"] - prev_mean) / prev_std
    out["share_up_D"] = (price > day_mean).astype(float).groupby(day).transform("mean").shift(LAG1)

    # time of day / calendar of day D+1 (allowed by the spec)
    out["sin_period"] = np.sin(2 * np.pi * out["period"] / N)
    out["cos_period"] = np.cos(2 * np.pi * out["period"] / N)
    dow = day.day_of_week
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(float)
    out["is_holiday"] = _nsw_holidays(day).astype(float)

    # weather of day D+1 (all 12 columns, e.g. syd_temp, mel_solar)
    for c in WEATHER_COLS:
        out[c] = df[c]

    # calendar of day D+1: day-of-week one-hot and time of year
    for d in range(7):
        out[f"dow_{d}"] = (dow == d).astype(float)
    day_of_year = day.day_of_year
    out["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # interactions: day-D levels and D+1 weather modulated by time of day / year
    for scale in ("period", "year"):
        for name in ("demand", "price"):
            for wave in ("sin", "cos"):
                out[f"{name}_{wave}_{scale}"] = out[f"{name}_lag1"] * out[f"{wave}_{scale}"]
    out["temp_squared"] = out["syd_temp"] ** 2
    out["temp_sin_period"] = out["syd_temp"] * out["sin_period"]
    out["temp_cos_period"] = out["syd_temp"] * out["cos_period"]
    out["price_relative_mean"] = out["price_lag1"] - prev_mean     # $/MWh, unscaled
    out["demand_pct_change"] = out["demand_change"] / out["demand_lag7"].clip(lower=1)
    out["solar_sin_period"] = out["syd_solar"] * out["sin_period"]
    out["solar_cos_period"] = out["syd_solar"] * out["cos_period"]

    # interactions with day D's price direction
    for scale in ("period", "year"):
        for wave in ("sin", "cos"):
            out[f"up_lag1_{wave}_{scale}"] = out["up_lag1"] * out[f"{wave}_{scale}"]
    out["up_lag1_lag7"] = out["up_lag1"] * out["up_lag7"]
    out["up_lag1_price_relative"] = out["up_lag1"] * out["price_relative_mean"]
    for d in range(1, 7):
        out[f"up_lag1_dow_{d}"] = out["up_lag1"] * out[f"dow_{d}"]
    return out


def _feature_names(t):
    return [c for c in t.columns if c not in META + TARGETS]


def build_table(df):
    """All rows with every feature and target defined (drops the first 7 days)."""
    t = _build(df.sort_index())
    return t.dropna(subset=_feature_names(t) + TARGETS)


def features_for_day(history, weather_next):
    """The 48 feature rows for day D+1, from the inputs Forecaster.predict receives."""
    tail = history.iloc[-8 * N:]                                  # days D-7..D covers every lag
    nxt = weather_next.reindex(columns=history.columns)           # market columns -> NaN
    t = _build(pd.concat([tail, nxt]))
    return t.loc[weather_next.index, _feature_names(t)]


# ---- self-test -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    df = load(sys.argv[1])
    t = build_table(df)
    train, valid = stage1_masks(t)
    print(f"table: {t.shape[0]:,} rows, {len(_feature_names(t))} features defined, "
          f"{t['day'].min().date()} to {t['day'].max().date()}")
    print(f"stage 1: train {train.sum():,} rows (2010-13), valid {valid.sum():,} rows (2014)")

    days = market_day(df.index)
    rng = np.random.default_rng(0)
    for d in rng.choice(t["day"].unique(), 20, replace=False):
        d = pd.Timestamp(d)
        hist, wx = df[days < d], df.loc[days == d, WEATHER_COLS]
        got = features_for_day(hist, wx)
        want = t.loc[wx.index, got.columns]
        pd.testing.assert_frame_equal(got, want, check_dtype=False)
    print("parity OK: features_for_day == build_table on 20 random days")
