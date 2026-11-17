"""Compare every price-direction model in predictions/ against both baselines, on identical rows.

Run from the repo root:
    python classification/analysis.py [--section frozen] [--title "..."]

Prints the 2014 (Stage 1) table and frozen 2015-26 balanced accuracy by year, writes
figures/<section>_yearly_bal_acc.png and figures/<section>_monthly_accuracy_ci.png, and
records all of it as section <section> of RESULTS.md.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev

ap = argparse.ArgumentParser()
ap.add_argument("--section", default="frozen", help="RESULTS.md section key and figure prefix")
ap.add_argument("--title", default="Frozen models vs baselines")
args = ap.parse_args()

preds = ev.load_predictions()
ref = ev.labels_and_baselines(ev.load_df())

# every model must be scored on the same rows with the same labels
shared_idx = ref.index
for name, p in preds.items():
    shared_idx = shared_idx.intersection(p.index)
ref = ref.loc[shared_idx]
for name, p in preds.items():
    bad = (p.loc[shared_idx, "y_true"] != ref["y_true"]).sum()
    assert bad == 0, f"{name}: {bad} labels disagree with the shared label definition"

cols = {name: p.loc[shared_idx, "y_pred"] for name, p in preds.items()}
cols["baseline: yesterday"] = ref["yesterday"]
cols["baseline: last week"] = ref["last_week"]
year = ref["day"].dt.year


def scores(mask):
    yt = ref.loc[mask, "y_true"]
    return pd.DataFrame({n: {"Acc %": ev.da.accuracy(yt, c[mask]),
                             "BalAcc %": ev.da.balanced_accuracy(yt, c[mask])}
                         for n, c in cols.items()}).T.round(2)


models = pd.DataFrame({name: {"model": info.get("model", ""),
                              "params": ", ".join(f"{k}={v}" for k, v in info.get("params", {}).items()),
                              "features": info.get("n_features", "")}
                       for name, info in ((n, ev.load_info(n)) for n in preds)}).T
models.index.name = "predictions file"

val = scores(year == 2014)
val.index.name = "2014"
yearly = pd.DataFrame({y: {n: ev.da.balanced_accuracy(ref.loc[year == y, "y_true"], c[year == y])
                           for n, c in cols.items()}
                       for y in range(2015, 2027)}).T.round(2)
yearly.index.name = "year"
yearly["class-1 share"] = ref["y_true"].groupby(year).mean().loc[2015:2026].round(3)

print(f"{len(shared_idx):,} shared half-hours; models: {', '.join(preds)}\n")
print("Stage 1 validation (2014, models fitted on 2010-13)")
print(val.to_string(), "\n")
print("Frozen models, balanced accuracy % by year (fitted on 2010-14)")
print(yearly.to_string())

os.makedirs(ev.FIG_DIR, exist_ok=True)
fig_yearly = f"{args.section}_yearly_bal_acc.png"
fig_monthly = f"{args.section}_monthly_accuracy_ci.png"

ax = yearly.drop(columns="class-1 share").plot(marker="o", figsize=(9, 4.5))
for line in ax.get_lines():
    if line.get_label().startswith("baseline"):
        line.set_linestyle("--")
ax.set(xlabel="Year", ylabel="Balanced accuracy (%)",
       title="Frozen price-direction models vs baselines, 2015-2026")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ev.FIG_DIR, fig_yearly), dpi=150)

frozen = year >= 2015
fig, ax = plt.subplots(figsize=(11, 4.5))
for name in preds:
    correct = (cols[name][frozen] == ref.loc[frozen, "y_true"]).astype(int)
    ci = ev.monthly_accuracy_ci(ref.loc[frozen, "day"], correct)
    line, = ax.plot(ci.index, ci["acc"] * 100, label=name, lw=1.4)
    ax.fill_between(ci.index, ci["boot_lo"] * 100, ci["boot_hi"] * 100,
                    color=line.get_color(), alpha=0.18)
ax.set(xlabel="Month", ylabel="Accuracy (%)",
       title="Frozen models: monthly accuracy with 95% day-block bootstrap CI")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ev.FIG_DIR, fig_monthly), dpi=150)

command = "python classification/analysis.py" + (
    f" --section {args.section} --title \"{args.title}\"" if args.section != "frozen" else "")
body = f"""
Models ({len(shared_idx):,} shared half-hours, identical labels):

{ev.md_table(models)}

**Stage 1 validation**: fitted on 2010-13, scored on 2014.

{ev.md_table(val)}

**Frozen models**: fitted on 2010-14, balanced accuracy % by year.

{ev.md_table(yearly)}

![yearly balanced accuracy](figures/{fig_yearly})

![monthly accuracy with 95% CI](figures/{fig_monthly})
"""
ev.write_section(args.section, args.title, body, command)
