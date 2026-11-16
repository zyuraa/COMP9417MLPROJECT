# Price-direction classifiers: shared comparison

Every classifier is scored on **the same half-hours with the same labels**, and all of them
appear on the same tables and figures. All models build their inputs from one feature
builder and write one predictions file in a shared format.

```
classification/
  features.py      every feature, defined once (each model picks its own FEATURES list)
  evaluation.py    shared: data path, labels + baselines, save_predictions(), monthly CIs
  knn.py           kNN                  -> predictions/knn.parquet
  logreg.py        logistic regression  -> predictions/logreg.parquet
  analysis.py      reads every file in predictions/ -> tables + figures/
  predictions/     one parquet per model: y_true, y_pred, p_up, split (val 2014 / frozen 2015-26)
  figures/         yearly_bal_acc.png, monthly_accuracy_ci.png
```

Run from the repo root (Python 3.11 env from requirements.txt):

```
python classification/knn.py         # ~30 s
python classification/logreg.py      # ~10 s
python classification/analysis.py    # tables + figures
```

## Adding a model

Copy `logreg.py` or `knn.py`, set `FEATURES` (any columns defined in `features.py`; add new
features there) and `model()`, and change the name passed to `save_predictions`. Each script
fits on 2010-13 and predicts 2014 (`val`), then refits on 2010-14 and predicts 2015-26
(`frozen`).

`logreg.py` is the logistic regression from `starter_notebook.ipynb` (same features, same
column order, same model); its predictions match the notebook exactly (2014 BA 79.28).

analysis.py checks that every model's `y_true` matches the shared label definition
(price above day D's mean, spec p.3), so a mismatched pipeline fails loudly instead of
plotting silently wrong numbers.
