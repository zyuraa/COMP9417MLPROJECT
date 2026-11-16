"""Shared plumbing for the price-direction classifiers, so every model is compared on the
same rows, the same labels and the same plots.

The contract: each model writes ONE predictions file with save_predictions(name, ...):

    classification/predictions/<name>.parquet
        index   timestamp (end of half-hour, as in the CSV)
        y_true  0/1 price-direction label (spec p.3: price above day D's mean)
        y_pred  0/1 prediction
        p_up    optional, predicted probability of class 1 (NaN if the model has none)
        split   "val" (2014, model fitted on 2010-13) or "frozen" (2015-26, fitted on 2010-14)

analysis.py then reads every file in predictions/ and puts all models + both baselines
on the same tables and figures. A model never has to know about the others.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import dayahead as da  # the marking metrics

DATA = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
PRED_DIR = os.path.join(ROOT, "classification", "predictions")
FIG_DIR = os.path.join(ROOT, "classification", "figures")


def load_df():
    return da.load_dataset(DATA)


def labels_and_baselines(df):
    """y_true plus both spec baselines for every half-hour (same definitions as the notebook)."""
    day = da.market_day(df.index)
    mean_d = df.groupby(day)["nsw_price"].transform("mean").shift(48)   # mean price of day D
    out = pd.DataFrame(index=df.index)
    out["day"] = day
    out["y_true"] = (df["nsw_price"] > mean_d).astype(int)
    out["yesterday"] = (df["nsw_price"].shift(48) > mean_d).astype(int)
    out["last_week"] = (df["nsw_price"].shift(48 * 7) > mean_d).astype(int)
    return out.iloc[48 * 7:]


def save_predictions(name, index, y_true, y_pred, p_up=None):
    """Write one model's predictions in the shared format (see module docstring)."""
    os.makedirs(PRED_DIR, exist_ok=True)
    idx = pd.DatetimeIndex(index, name="timestamp")
    out = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "p_up": np.nan if p_up is None else np.asarray(p_up, dtype=float),
    }, index=idx)
    year = da.market_day(idx).year
    out["split"] = np.where(year == 2014, "val", np.where(year >= 2015, "frozen", "train"))
    out = out[out["split"] != "train"]
    path = os.path.join(PRED_DIR, f"{name}.parquet")
    out.to_parquet(path)
    print(f"saved {len(out):,} predictions -> {os.path.relpath(path, ROOT)}")
    return path


def load_predictions():
    """{model name: predictions frame} for every file in predictions/."""
    out = {}
    for f in sorted(os.listdir(PRED_DIR)):
        if f.endswith(".parquet"):
            out[f[:-8]] = pd.read_parquet(os.path.join(PRED_DIR, f))
    return out


def monthly_accuracy_ci(day, correct, n_boot=1000, seed=0):
    """Per month: accuracy, CI treating half-hours as independent, and a day-block bootstrap
    CI (spec p.5 item 2, Appendix A). `correct` is 1/0 per half-hour, `day` its market day."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"day": day, "correct": correct})
    frame["month"] = frame["day"].dt.to_period("M")
    rows = []
    for month, g in frame.groupby("month"):
        acc, n = g["correct"].mean(), len(g)
        se = np.sqrt(acc * (1 - acc) / n)
        per_day = g.groupby("day")["correct"].agg(["sum", "count"]).to_numpy()
        pick = rng.integers(0, len(per_day), size=(n_boot, len(per_day)))
        boots = per_day[pick, 0].sum(1) / per_day[pick, 1].sum(1)
        rows.append({"month": month.to_timestamp(), "acc": acc,
                     "ind_lo": acc - 1.96 * se, "ind_hi": acc + 1.96 * se,
                     "boot_lo": np.quantile(boots, 0.025), "boot_hi": np.quantile(boots, 0.975)})
    return pd.DataFrame(rows).set_index("month")
