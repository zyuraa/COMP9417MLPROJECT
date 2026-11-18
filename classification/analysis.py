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
                              "features": info.get("n_features", ""),
                              "retrains": ", ".join(info.get("strategy", {}).get("retrains", []))
                                          or "none (frozen)"}
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
print("2015-26 replay, balanced accuracy % by year (frozen = fitted on 2010-14, others retrained per strategy)")
print(yearly.to_string())

os.makedirs(ev.FIG_DIR, exist_ok=True)
fig_yearly = f"{args.section}_yearly_bal_acc.png"
fig_monthly = f"{args.section}_monthly_accuracy_ci.png"

BASELINE_STYLE = {"baseline: yesterday": "#8a8a8a", "baseline: last week": "#bdbdbd"}
STRATEGY_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def strategy(info):
    """Short label for a predictions file's retraining strategy."""
    s = info.get("strategy")
    if not s:
        return "frozen"
    when = f"every {s['every_years']}y" if s.get("every_years") else "drift alarms"
    window = f"sliding {s['window_years']}y" if s.get("window_years") else "expanding"
    return f"{when}, {window}"


# one panel per model family; within a panel, one line per strategy (same colour in every panel)
infos = {n: ev.load_info(n) for n in preds}
families = list(dict.fromkeys(i.get("model", n) for n, i in infos.items()))
labels = list(dict.fromkeys(strategy(i) for i in infos.values()))
colour = dict(zip(labels, STRATEGY_COLOURS))
panel = {f: [n for n, i in infos.items() if i.get("model", n) == f] for f in families}

fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families) + 2, 4.5),
                         sharey=True, squeeze=False)
for ax, fam in zip(axes[0], families):
    for name in panel[fam]:
        lab = strategy(infos[name])
        ax.plot(yearly.index, yearly[name], marker="o", lw=2, color=colour[lab], label=lab)
    for name, c in BASELINE_STYLE.items():  # baselines: grey, dashed, behind the models
        ax.plot(yearly.index, yearly[name], ls="--", lw=1.2, color=c, label=name, zorder=1)
    ax.set(xlabel="Year", title=fam)
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("Balanced accuracy (%)")
fig.suptitle(f"{args.title}: balanced accuracy by year")
handles = {l: h for ax in axes[0] for h, l in zip(*ax.get_legend_handles_labels())}
fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), fontsize=9,
           frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(os.path.join(ev.FIG_DIR, fig_yearly), dpi=150)

frozen = year >= 2015
fig, axes = plt.subplots(len(families), 1, figsize=(11, 3.6 * len(families)),
                         sharex=True, sharey=True, squeeze=False)
for ax, fam in zip(axes[:, 0], families):
    for name in panel[fam]:
        lab = strategy(infos[name])
        correct = (cols[name][frozen] == ref.loc[frozen, "y_true"]).astype(int)
        ci = ev.monthly_accuracy_ci(ref.loc[frozen, "day"], correct)
        ax.plot(ci.index, ci["acc"] * 100, label=lab, lw=1.4, color=colour[lab])
        ax.fill_between(ci.index, ci["boot_lo"] * 100, ci["boot_hi"] * 100,
                        color=colour[lab], alpha=0.18)
    ax.set(ylabel="Accuracy (%)", title=fam)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
axes[-1][0].set_xlabel("Month")
fig.suptitle("Monthly accuracy with 95% day-block bootstrap CI")
plt.tight_layout()
plt.savefig(os.path.join(ev.FIG_DIR, fig_monthly), dpi=150)

command = "python classification/analysis.py" + (
    f" --section {args.section} --title \"{args.title}\"" if args.section != "frozen" else "")
body = f"""
Models ({len(shared_idx):,} shared half-hours, identical labels):

{ev.md_table(models)}

**Stage 1 validation**: fitted on 2010-13, scored on 2014.

{ev.md_table(val)}

**2015-26 replay**: balanced accuracy % by year. Models without a strategy are frozen (fitted on 2010-14); retrained ones list their dates in the models table.

{ev.md_table(yearly)}

![yearly balanced accuracy](figures/{fig_yearly})

![monthly accuracy with 95% CI](figures/{fig_monthly})
"""
ev.write_section(args.section, args.title, body, command)
