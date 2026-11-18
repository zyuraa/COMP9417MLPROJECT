import pandas as pd
import matplotlib.pyplot as plt

from drift import market_day, build_error_detector

# 1. Load model predictions
log_pred = pd.read_parquet(
    "classification/predictions/logreg.parquet"
)

knn_pred = pd.read_parquet(
    "classification/predictions/knn.parquet"
)

# 2. Calculate daily classification error
def get_daily_error(pred):
    errors = (
        pred["y_true"] != pred["y_pred"]
    ).astype(float)

    return errors.groupby(
        market_day(pred.index)
    ).mean()

log_daily_err = get_daily_error(log_pred)
knn_daily_err = get_daily_error(knn_pred)

# 3. Create CUSUM detectors
log_detector = build_error_detector(
    log_daily_err,
    cooldown=60,
    max_alarms=6
)

knn_detector = build_error_detector(
    knn_daily_err,
    cooldown=60,
    max_alarms=6
)

# 4. Run online detection (2015–2026)
log_path, log_alarms = log_detector.run(
    log_daily_err.loc["2015-01-01":]
)

knn_path, knn_alarms = knn_detector.run(
    knn_daily_err.loc["2015-01-01":]
)

print("Logistic Regression alarms:", log_alarms)
print("KNN alarms:", knn_alarms)

# 5. Plot CUSUM statistics
fig, axes = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True
)

for ax, path, detector, alarms, name in [
    (axes[0], log_path, log_detector,
     log_alarms, "Logistic Regression"),
    (axes[1], knn_path, knn_detector,
     knn_alarms, "KNN")
]:
    ax.plot(path.index, path, label="CUSUM")

    ax.axhline(
        detector.h,
        color="red",
        linestyle="--",
        label="Threshold"
    )

    for date in alarms:
        ax.axvline(date, color="red", alpha=0.3)

    ax.set_title(name)
    ax.set_ylabel("CUSUM Statistic")
    ax.legend()
    ax.grid(alpha=0.3)

plt.xlabel("Date")
plt.tight_layout()
plt.show()