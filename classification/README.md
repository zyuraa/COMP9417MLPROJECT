# Price-direction classifiers: shared comparison

Every classifier is scored on **the same half-hours with the same labels**, and all of them
appear on the same tables and figures. All models build their inputs from one feature
builder and write one predictions file in a shared format.

```
classification/
  features.py      every feature, defined once; SHARED_FEATURES (47) and FEATURE_GROUPS
  evaluation.py    shared: data path, labels + baselines, save_predictions(), monthly CIs
  knn.py           kNN (selected model)  -> predictions/knn.parquet
  logreg.py        logistic regression   -> predictions/logreg.parquet
  analysis.py      reads every file in predictions/ -> tables + figures/<section>_*.png
  stage1/          model and feature selection on 2014 (concluded, see RESULTS.md)
    tune.py         hyperparameter grid for both models
    importance.py   drop-group refit: what each feature group adds
  RESULTS.md       every table and figure, with the command that produced it
  predictions/     one parquet per model (y_true, y_pred, p_up, split) + a json of its settings
  figures/         all figures, named after the script or section that wrote them
```

Run from the repo root (Python 3.11 env from requirements.txt):

```
python classification/knn.py                              # ~30 s
python classification/logreg.py --class-weight balanced   # ~10 s
python classification/analysis.py                         # tables + figures
```

Stage 1 experiments (only needed to reproduce the selection):

```
python classification/stage1/tune.py         # ~2 min
python classification/stage1/importance.py   # ~1 min
```

## Adding a model

Copy `logreg.py` or `knn.py`, set `FEATURES` (any columns defined in `features.py`; add new
features there) and `model()`, and change the name passed to `save_predictions`. Each script
fits on 2010-13 and predicts 2014 (`val`), then refits on 2010-14 and predicts 2015-26
(`frozen`).

`logreg.py` is the logistic regression from `starter_notebook.ipynb` (same features, same
column order, same model); on the notebook's 46 features its predictions match the notebook
exactly (2014 BA 79.28). It now uses the shared set, which adds `is_holiday`.

analysis.py checks that every model's `y_true` matches the shared label definition
(price above day D's mean, spec p.3), so a mismatched pipeline fails loudly instead of
plotting silently wrong numbers.

## Results doc

Scripts record their output in `RESULTS.md` through `evaluation.write_section`. Each section
is between `<!-- BEGIN key -->` and `<!-- END key -->` markers and starts with the exact
command and code version that produced it. Rerunning the command replaces only that section,
so notes written outside the markers are kept. `analysis.py --section <key> --title "..."`
records a new snapshot (e.g. before and after a change) without overwriting older ones.
