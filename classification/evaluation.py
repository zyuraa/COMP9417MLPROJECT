"""Shared plumbing for the price-direction classifiers, so every model is compared on the
same rows, the same labels and the same plots.

The contract: each model writes ONE predictions file with save_predictions(name, ...):

    classification/predictions/<name>.parquet
        index   timestamp (end of half-hour, as in the CSV)
        y_true  0/1 price-direction label (spec p.3: price above day D's mean)
        y_pred  0/1 prediction
        p_up    optional, predicted probability of class 1 (NaN if the model has none)
        split   "val" (2014, model fitted on 2010-13) or "frozen" (2015-26, fitted on 2010-14)

    classification/predictions/<name>.json     settings the predictions came from (info=...)

analysis.py then reads every file in predictions/ and puts all models + both baselines
on the same tables and figures. A model never has to know about the others.

Results doc: scripts record their tables and figures in classification/RESULTS.md with
write_section(key, title, body). Each key owns the text between its two markers, so a rerun
replaces only that section and hand-written notes outside the markers are kept.
"""
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import dayahead as da  # the marking metrics

DATA = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
PRED_DIR = os.path.join(ROOT, "classification", "predictions")
FIG_DIR = os.path.join(ROOT, "classification", "figures")
RESULTS = os.path.join(ROOT, "classification", "RESULTS.md")


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


def save_predictions(name, index, y_true, y_pred, p_up=None, info=None):
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
    with open(os.path.join(PRED_DIR, f"{name}.json"), "w") as f:
        json.dump({"name": name, "created": stamp(), **(info or {})}, f, indent=2)
    print(f"saved {len(out):,} predictions -> {os.path.relpath(path, ROOT)}")
    return path


def load_info(name):
    path = os.path.join(PRED_DIR, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else {"name": name}


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


# ---- results doc -----------------------------------------------------------------------------

def stamp():
    """Date plus code version, so every recorded number can be traced back."""
    try:
        rev = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = "unknown"
    return f"{datetime.now():%Y-%m-%d %H:%M}, code {rev or 'unknown'}"


def md_table(frame, floatfmt="{}"):
    """A DataFrame as a Markdown table (index included)."""
    cols = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_section(key, title, body, command):
    """Create or replace the section `key` in RESULTS.md."""
    start, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
    block = (f"{start}\n## {title}\n\n`{command}` ({stamp()})\n\n{body.strip()}\n{end}")
    doc = open(RESULTS).read() if os.path.exists(RESULTS) else (
        "# Price-direction results\n\nEach section below is written by the script named in it; "
        "rerun the command to regenerate it. Notes outside the BEGIN/END markers are kept.\n")
    if start in doc and end in doc:
        doc = doc[:doc.index(start)] + block + doc[doc.index(end) + len(end):]
    else:
        doc = doc.rstrip() + "\n\n" + block + "\n"
    with open(RESULTS, "w") as f:
        f.write(doc)
    print(f"RESULTS.md: section '{key}' updated")
