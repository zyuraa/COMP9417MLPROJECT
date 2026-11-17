"""Stage 1 hyperparameter search for the price-direction classifiers, scored on 2014.

Every setting is fitted on 2010-2013 and scored on 2014 (balanced accuracy), using the
same features as knn.py and logreg.py. Writes figures/tuning.png and the "tuning" section
of RESULTS.md, and prints the best setting per model.

Run from the repo root:
    python classification/stage1/tune.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation as ev
import features as F

FEATURES = F.SHARED_FEATURES
GRIDS = {
    "kNN": [{"k": k, "weights": w}
            for w in ("uniform", "distance") for k in (5, 10, 20, 35, 50, 75, 100, 150, 200, 300)],
    "logistic regression": [{"C": c, "class_weight": cw}
                            for cw in ("none", "balanced")
                            for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
}


def build(model, p):
    if model == "kNN":
        est = KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"], n_jobs=-1)
    else:
        est = LogisticRegression(C=p["C"], max_iter=1000,
                                 class_weight=None if p["class_weight"] == "none" else "balanced")
    return make_pipeline(StandardScaler(), est)


t = F.build_table(F.load(ev.DATA))
X, y = t[FEATURES], t["y_up"].astype(int)
train, valid = (t["year"] <= 2013).to_numpy(), (t["year"] == 2014).to_numpy()

rows = []
for model, grid in GRIDS.items():
    for p in grid:
        start = time.time()
        pred = build(model, p).fit(X[train], y[train]).predict(X[valid])
        rows.append({"model": model, **p,
                     "acc": round(ev.da.accuracy(y[valid], pred), 2),
                     "bal_acc": round(ev.da.balanced_accuracy(y[valid], pred), 2)})
        print(f"{model:20s} {p}  BalAcc {rows[-1]['bal_acc']:.2f}  ({time.time() - start:.0f}s)")
res = pd.DataFrame(rows)

knn = res[res["model"] == "kNN"]
lr = res[res["model"] == "logistic regression"]
best_knn, best_lr = knn.loc[knn["bal_acc"].idxmax()], lr.loc[lr["bal_acc"].idxmax()]
default_knn = knn[(knn["k"] == 50) & (knn["weights"] == "distance")].iloc[0]
default_lr = lr[(lr["C"] == 1.0) & (lr["class_weight"] == "none")].iloc[0]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for w, g in knn.groupby("weights"):
    a1.plot(g["k"], g["bal_acc"], marker="o", label=f"weights={w}")
a1.set(xscale="log", xlabel="k (neighbours, log scale)", ylabel="2014 balanced accuracy (%)",
       title="kNN")
for cw, g in lr.groupby("class_weight"):
    a2.plot(g["C"], g["bal_acc"], marker="o", label=f"class_weight={cw}")
a2.set(xscale="log", xlabel="C (inverse regularisation, log scale)", title="Logistic regression")
for ax in (a1, a2):
    ax.axhline(ev.da.balanced_accuracy(y[valid], t.loc[valid, "up_lag1"].astype(int)),
               color="grey", ls="--", lw=1, label="baseline: yesterday")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
os.makedirs(ev.FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(ev.FIG_DIR, "tuning.png"), dpi=150)

summary = pd.DataFrame({
    "kNN": {"before": f"k=50, distance: {default_knn['bal_acc']:.2f}",
            "best": f"k={int(best_knn['k'])}, {best_knn['weights']}: {best_knn['bal_acc']:.2f}"},
    "logistic regression": {
        "before": f"C=1, class_weight=none: {default_lr['bal_acc']:.2f}",
        "best": f"C={best_lr['C']}, class_weight={best_lr['class_weight']}: {best_lr['bal_acc']:.2f}"},
}).T
summary.index.name = "model"
grid_knn = knn.astype({"k": int}).pivot(index="k", columns="weights", values="bal_acc")
grid_lr = lr.pivot(index="C", columns="class_weight", values="bal_acc")
body = f"""
Fitted on 2010-13, scored on 2014; {len(FEATURES)} shared features. Balanced accuracy %.

{ev.md_table(summary)}

kNN (rows k, columns weighting):

{ev.md_table(grid_knn)}

Logistic regression (rows C, columns class weighting):

{ev.md_table(grid_lr)}

![tuning curves](figures/tuning.png)
"""
ev.write_section("tuning", "Stage 1 hyperparameter search (2014)", body,
                 "python classification/stage1/tune.py")
print("\n" + summary.to_string())
