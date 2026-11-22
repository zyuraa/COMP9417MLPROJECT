"""Monthly accuracy of a frozen price-direction model over 2015-2026 with both 95% CIs
(spec p.5, Performance Degradation item 1-2, and Appendix A):

  independent : each half-hour treated as an independent Bernoulli trial (normal approximation)
  day-block   : bootstrap that resamples whole days, keeping within-day correlation

The same-half-hour-yesterday baseline's monthly accuracy is drawn for reference.
Writes figures/frozen_ci_<name>.png and the "frozen_ci_<name>" section of RESULTS.md.
Run from the repo root:
    python classification/stage2/frozen_ci.py [--name knn] [--label "frozen kNN"]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation as ev

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="knn", help="predictions file (must be a frozen model)")
ap.add_argument("--label", default="frozen kNN (fitted on 2010-14)")
args = ap.parse_args()

pred = pd.read_parquet(os.path.join(ev.PRED_DIR, f"{args.name}.parquet"))
pred = pred[pred["split"] == "frozen"]
ref = ev.labels_and_baselines(ev.load_df()).loc[pred.index]
assert (ref["y_true"].to_numpy() == pred["y_true"].to_numpy()).all(), "labels disagree"

ci = ev.monthly_accuracy_ci(ref["day"], (pred["y_pred"] == pred["y_true"]).astype(int)) * 100
base = (ref["yesterday"] == ref["y_true"]).groupby(ref["day"].dt.to_period("M")).mean() * 100
base.index = base.index.to_timestamp()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(ci.index, ci["boot_lo"], ci["boot_hi"], color="#f4a261", alpha=0.35,
                label="95% CI, day-block bootstrap")
ax.fill_between(ci.index, ci["ind_lo"], ci["ind_hi"], color="#1f77b4", alpha=0.45,
                label="95% CI, independent half-hours")
ax.plot(ci.index, ci["acc"], color="black", lw=1.3, label=args.label)
ax.plot(base.index, base, color="#8a8a8a", ls="--", lw=1, label="baseline: same half-hour yesterday")
ax.set(xlabel="Month", ylabel="Accuracy (%)",
       title=f"Monthly accuracy of the {args.label}, Jan 2015 - Oct 2026, with 95% CIs")
ax.legend(fontsize=9, loc="lower left", ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
fig_name = f"frozen_ci_{args.name}.png"
plt.savefig(os.path.join(ev.FIG_DIR, fig_name), dpi=150)

width_ind = (ci["ind_hi"] - ci["ind_lo"]).mean()
width_boot = (ci["boot_hi"] - ci["boot_lo"]).mean()
below = (ci["acc"] < base.reindex(ci.index)).sum()
yearly = ci["acc"].groupby(ci.index.year).mean().round(2).rename("mean monthly accuracy %")
yearly.index.name = "year"
print(f"mean CI width: independent {width_ind:.2f} pp, day-block {width_boot:.2f} pp "
      f"({width_boot / width_ind:.1f}x); months below the yesterday baseline: {below}/{len(ci)}")

body = f"""
Monthly accuracy (%) of `{args.name}` over the frozen replay, with 95% CIs computed both ways
(Appendix A). Mean CI width: **{width_ind:.2f} pp** treating half-hours as independent,
**{width_boot:.2f} pp** with the day-block bootstrap (**{width_boot / width_ind:.1f}x** wider),
because errors within a day are correlated. The model is below the yesterday baseline in
{below} of {len(ci)} months.

![frozen monthly accuracy with both CIs](figures/{fig_name})

{ev.md_table(yearly.to_frame())}
"""
ev.write_section(f"frozen_ci_{args.name}", f"Frozen monthly accuracy with 95% CIs: {args.name}",
                 body, f"python classification/stage2/frozen_ci.py --name {args.name}")
