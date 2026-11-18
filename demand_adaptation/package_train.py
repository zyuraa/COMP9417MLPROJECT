# Rebuild the selected demand model from the released CSV and locked settings

import argparse
import json
from pathlib import Path

import pandas as pd


def save_artifacts(data, model, settings, folder):
    folder.mkdir(parents=True, exist_ok=True)
    data.save(model, folder / "demand_model.joblib")
    metadata = {key: settings[key] for key in ("selected_strategy", "train_start", "train_end")}
    metadata["window_days"] = 7
    (folder / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train(csv):
    from lasso import prepare_lasso
    folder = Path(__file__).resolve().parent
    config = folder / "settings.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    df = pd.read_csv(csv, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    cutoff = pd.Timestamp(settings["train_end"]) + pd.Timedelta(days=1)
    if df.index[-1] < cutoff:
        raise ValueError("The CSV ends before the configured final training date.")
    data = prepare_lasso(df.loc[:cutoff], transformer_year=settings["reference_transformer_through"],
                         config_path=config)
    model = data.fit(settings["train_start"], settings["train_end"])
    save_artifacts(data, model, settings, folder / "artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    train(parser.parse_args().csv)
