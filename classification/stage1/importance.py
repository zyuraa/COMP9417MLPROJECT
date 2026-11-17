"""Feature-group importance on 2014 (drop-group refit), for both models.

Each feature group is removed in turn, the model is retrained on 2010-2013 without it, and
the loss in 2014 balanced accuracy against the full model is recorded. Groups are removed
rather than single features because correlated features cover for each other.

Writes figures/importance.png and the "importance" section of RESULTS.md.
Run from the repo root:
    python classification/stage1/importance.py
"""
import os
import sys

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
GROUPS = F.FEATURE_GROUPS
MODELS = {
    "kNN": lambda: KNeighborsClassifier(50, weights="distance", n_jobs=-1),
    "logreg": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000),
}

t = F.build_table(F.load(ev.DATA))
X, y = t[FEATURES], t["y_up"].astype(int)
train, valid = (t["year"] <= 2013).to_numpy(), (t["year"] == 2014).to_numpy()


def score(make, cols):
    model = make_pipeline(StandardScaler(), make()).fit(X.loc[train, cols], y[train])
    return ev.da.balanced_accuracy(y[valid], model.predict(X.loc[valid, cols]))


refit = {}
for name, make in MODELS.items():
    base = score(make, FEATURES)
    refit[name] = {g: base - score(make, [f for f in FEATURES if f not in cols])
                   for g, cols in GROUPS.items()}
    print(f"{name}: base 2014 BA {base:.2f}")
refit = pd.DataFrame(refit).round(2)
refit["features"] = [len(GROUPS[g]) for g in refit.index]
refit.index.name = "group removed"
refit = refit.sort_values("kNN", ascending=False)

ax = refit[list(MODELS)].iloc[::-1].plot.barh(figsize=(8, 4.5))
ax.set(xlabel="loss in 2014 balanced accuracy (pp) when retrained without the group", ylabel="",
       title="Drop-group refit: what each feature group adds")
ax.axvline(0, color="black", lw=0.8)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
os.makedirs(ev.FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(ev.FIG_DIR, "importance.png"), dpi=150)

body = f"""
Both models fitted on 2010-13 ({len(FEATURES)} shared features). Each group is removed in
turn, the model is retrained without it, and the loss in 2014 balanced accuracy against the
full model is recorded (percentage points; near zero or negative = no measurable
contribution). Groups are removed rather than single features because correlated features
cover for each other: removing one lets the model lean on its near-duplicate.

{ev.md_table(refit)}

![drop-group refit](figures/importance.png)
"""
ev.write_section("importance", "Feature-group importance: drop-group refit (2014)", body,
                 "python classification/stage1/importance.py")
print("\n" + refit.to_string())
