"""
drift.py - helpers for the "Performance Degradation and Drift Detection" section.

Model-agnostic: nothing here needs a fitted model. Everything works on
  (a) the raw half-hourly CSV (input-based detector), and
  (b) a half-hourly prediction table with columns y_true, y_pred, c_true, c_pred
      indexed by timestamp (same format run_baselines.py already saves).

Sections
  1. Monthly error + 95% intervals (independent / Hoeffding / day-block bootstrap)
  2. Input-based drift score (deseasonalised daily features)
  3. CUSUM detector engine (used for both the input and the error detector)
  4. Comparing detections with observed degradation
All detectors are ONLINE: update() sees one day at the end of day D only.
"""
import numpy as np
import pandas as pd

N_PERIODS = 48


def market_day(index):
    """Timestamps are interval ENDS: D 00:30 ... D+1 00:00 belong to day D."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)).normalize()


# ============================================================================
# 1. Monthly errors with 95% intervals (Appendix A)
# ============================================================================
def day_block_bootstrap(errors_by_day, stat=np.mean, n_boot=1000, seed=0):
    """Appendix A implementation (entries may also be multi-day chunks)."""
    rng = np.random.default_rng(seed)
    k = len(errors_by_day)
    stats = [stat(np.concatenate([errors_by_day[j] for j in rng.integers(0, k, k)]))
             for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def halfhour_errors(pred):
    """pred: index=timestamp, cols y_true, y_pred, c_true, c_pred.
    Returns per-half-hour ape (%) and correct (0/1) plus the market day."""
    out = pd.DataFrame(index=pred.index)
    out["day"] = market_day(pred.index)
    out["ape"] = 100.0 * (pred["y_true"] - pred["y_pred"]).abs() / pred["y_true"]
    out["correct"] = (pred["c_true"] == pred["c_pred"]).astype(float)
    return out


def _chunks(day_arrays, block_len):
    if block_len <= 1:
        return day_arrays
    return [np.concatenate(day_arrays[i:i + block_len])
            for i in range(0, len(day_arrays), block_len)]


def monthly_intervals(err, metric="ape", block_len=1, n_boot=1000, seed=0):
    """
    Monthly value + CIs. metric='ape' (MAPE, %) or 'correct' (accuracy, %).
      ind_*  : normal approximation, half-hours treated as independent
      hoef_* : Hoeffding (accuracy only)
      boot_* : day-block bootstrap (block_len=7 gives week blocks)
    """
    scale = 100.0 if metric == "correct" else 1.0
    rows = []
    for month, g in err.groupby(err["day"].dt.to_period("M")):
        days = [a.to_numpy() for _, a in g.groupby("day")[metric]]
        x = g[metric].to_numpy()
        n, m = len(x), x.mean()
        if metric == "correct":
            hw = 1.96 * np.sqrt(m * (1 - m) / n)
            hoef = np.sqrt(np.log(2 / 0.05) / (2 * n))
        else:
            hw = 1.96 * x.std(ddof=1) / np.sqrt(n)
            hoef = np.nan
        lo, hi = day_block_bootstrap(_chunks(days, block_len), n_boot=n_boot, seed=seed)
        rows.append(dict(month=month.to_timestamp(), n_halfhours=n, n_days=len(days),
                         value=scale * m,
                         ind_lo=scale * (m - hw), ind_hi=scale * (m + hw),
                         hoef_lo=scale * (m - hoef), hoef_hi=scale * (m + hoef),
                         boot_lo=scale * lo, boot_hi=scale * hi))
    t = pd.DataFrame(rows)
    t["width_ind"] = t["ind_hi"] - t["ind_lo"]
    t["width_boot"] = t["boot_hi"] - t["boot_lo"]
    t["width_ratio"] = t["width_boot"] / t["width_ind"]  # >1 => within-day correlation matters
    return t


def month_difference_ci(err, month_a, month_b, metric="ape", n_boot=2000, seed=0):
    """Bootstrap CI for (month_b - month_a): the proper test of 'did it change',
    unlike eyeballing overlap. Days are resampled independently inside each month."""
    rng = np.random.default_rng(seed)
    per = {}
    for name, mth in (("a", month_a), ("b", month_b)):
        g = err[err["day"].dt.to_period("M") == pd.Period(mth)]
        per[name] = [a.to_numpy() for _, a in g.groupby("day")[metric]]
    diffs = []
    for _ in range(n_boot):
        s = {k: np.concatenate([v[j] for j in rng.integers(0, len(v), len(v))])
             for k, v in per.items()}
        diffs.append(s["b"].mean() - s["a"].mean())
    scale = 100.0 if metric == "correct" else 1.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return scale * lo, scale * hi


def plot_monthly(table, ax, ylabel, title="", show_hoeffding=False):
    ax.fill_between(table["month"], table["boot_lo"], table["boot_hi"], alpha=0.3,
                    label="95% CI: day-block bootstrap")
    ax.fill_between(table["month"], table["ind_lo"], table["ind_hi"], alpha=0.5,
                    label="95% CI: independent half-hours")
    if show_hoeffding and table["hoef_lo"].notna().any():
        ax.plot(table["month"], table["hoef_lo"], "k:", lw=0.8, label="Hoeffding")
        ax.plot(table["month"], table["hoef_hi"], "k:", lw=0.8)
    ax.plot(table["month"], table["value"], lw=1.3, color="navy", label="Monthly value")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def daily_series(err, metric="ape"):
    """Daily mean MAPE (%) or daily ERROR RATE (1-accuracy, so 'higher = worse')."""
    s = err.groupby("day")[metric].mean()
    return s if metric == "ape" else 1.0 - s


# ============================================================================
# 2. Input-based drift score
# ============================================================================
def daily_input_features(df):
    """One row per complete market day, built from INPUT variables only.
    df: half-hourly CSV indexed by timestamp. Day D's features are known at end of D."""
    idx = df.index
    day = market_day(idx)
    shifted = idx - pd.Timedelta(minutes=30)
    slot = shifted.hour * 2 + shifted.minute // 30
    dem = pd.DataFrame({"day": day, "slot": slot, "v": df["nsw_demand"].to_numpy()}) \
        .pivot(index="day", columns="slot", values="v").dropna()
    f = pd.DataFrame(index=dem.index)
    m = dem.mean(axis=1)
    f["nsw_dem_mean"] = m                                    # level
    f["midday_ratio"] = dem.loc[:, 22:28].mean(axis=1) / m   # 11:00-14:30 (rooftop solar)
    f["evening_ratio"] = dem.loc[:, 34:40].mean(axis=1) / m  # 17:00-20:30 peak
    f["night_ratio"] = dem.loc[:, 2:8].mean(axis=1) / m      # 01:00-04:30 base
    g = df.groupby(day)
    for col in ["vic_demand", "qld_demand", "syd_temp", "syd_solar", "syd_cloud", "syd_rh"]:
        if col in df:
            f[col] = g[col].mean().reindex(f.index)
    f["nsw_price_med"] = g["nsw_price"].median().reindex(f.index)  # robust to spikes
    return f


def _design(index, holidays=None):
    doy = index.dayofyear.to_numpy()
    cols = [np.ones(len(index))]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    dow = index.dayofweek.to_numpy()
    cols += [(dow == d).astype(float) for d in range(1, 7)]
    hol = np.array([d.date() in holidays for d in index], float) if holidays else np.zeros(len(index))
    cols.append(hol)
    return np.column_stack(cols)


class InputScore:
    """Removes annual seasonality, day-of-week and holiday effects (fitted on the
    reference period only), standardises, and returns a daily drift score
    = mean over features of clip(z, +-5)^2   (about 1 when nothing has changed)."""

    def __init__(self, clip=5.0, holidays=None):
        self.clip, self.holidays = clip, holidays

    def fit(self, feats_ref):
        X = _design(feats_ref.index, self.holidays)
        self.cols_ = list(feats_ref.columns)
        self.beta_ = np.linalg.lstsq(X, feats_ref.to_numpy(), rcond=None)[0]
        resid = feats_ref.to_numpy() - X @ self.beta_
        self.sd_ = resid.std(axis=0, ddof=X.shape[1])
        return self

    def zscores(self, feats):
        X = _design(feats.index, self.holidays)
        z = (feats[self.cols_].to_numpy() - X @ self.beta_) / self.sd_
        return pd.DataFrame(np.clip(z, -self.clip, self.clip), index=feats.index, columns=self.cols_)

    def score(self, feats):
        return (self.zscores(feats) ** 2).mean(axis=1)


# ============================================================================
# 3. CUSUM engine (one-sided, detects an INCREASE)
# ============================================================================
class CUSUMDetector:
    """
    Statistic : S_t = max(0, S_{t-1} + (x_t - mu)/sigma - k)
    Reference : mu, sigma from fit_reference(x_ref)
    Threshold : h, from calibrate(x_cal) = margin * max of the S path on a
                calibration period (no resets), i.e. zero false alarms there.
    Online    : update(x) is called once per day, at the end of day D.
    cooldown/max_alarms keep detector-triggered retraining within the budget (6).
    """

    def __init__(self, k=0.5, cooldown=60, max_alarms=6):
        self.k, self.cooldown, self.max_alarms = k, cooldown, max_alarms
        self.h = None
        self.reset_state()

    def reset_state(self):
        self.S, self.since, self.n_alarms = 0.0, 10 ** 9, 0

    def fit_reference(self, x_ref):
        x = np.asarray(x_ref, float)
        self.mu, self.sigma = x.mean(), x.std(ddof=1)
        return self

    def _path(self, x):
        S, out = 0.0, []
        for v in np.asarray(x, float):
            S = max(0.0, S + (v - self.mu) / self.sigma - self.k)
            out.append(S)
        return np.array(out)

    def calibrate(self, x_cal, margin=1.0):
        self.h = margin * self._path(x_cal).max()
        return self

    def update(self, x):
        self.S = max(0.0, self.S + (x - self.mu) / self.sigma - self.k)
        self.since += 1
        if (self.S > self.h and self.since >= self.cooldown
                and self.n_alarms < self.max_alarms):
            self.S, self.since, self.n_alarms = 0.0, 0, self.n_alarms + 1
            return True
        return False

    def rebase(self, x_new_ref):
        """Call after a retrain: new reference stats (the model changed), S reset.
        Keep h in standardised units."""
        self.fit_reference(x_new_ref)
        self.S = 0.0

    def run(self, series):
        """Replay a daily series online. Returns (S path, alarm dates)."""
        self.reset_state()
        path, alarms = [], []
        for d, v in series.items():
            fired = self.update(v)
            path.append(self.S if not fired else self.h)
            if fired:
                alarms.append(d)
        return pd.Series(path, index=series.index), alarms


def build_input_detector(feats, ref=("2010-01-01", "2013-12-31"),
                         cal=("2014-01-01", "2014-12-31"), k=0.5, margin=1.1,
                         holidays=None, **kw):
    """Reference = 2010-2013 (fit seasonality + score mean/sd), threshold on 2014."""
    sc = InputScore(holidays=holidays).fit(feats.loc[ref[0]:ref[1]])
    score = sc.score(feats)
    det = CUSUMDetector(k=k, **kw).fit_reference(score.loc[ref[0]:ref[1]])
    det.calibrate(score.loc[cal[0]:cal[1]], margin=margin)
    return sc, det, score


def build_error_detector(daily_err, ref=("2014-01-01", "2014-12-31"),
                         cal=("2014-01-01", "2014-12-31"), k=0.5, margin=1.1, **kw):
    """daily_err: daily MAPE or daily error rate (higher = worse) of a FROZEN model.
    Defaults use 2014 (validation) for both reference and threshold, so the threshold
    covers one full seasonal cycle. Be explicit in the report that 2014 was also used
    for model selection."""
    det = CUSUMDetector(k=k, **kw).fit_reference(daily_err.loc[ref[0]:ref[1]])
    det.calibrate(daily_err.loc[cal[0]:cal[1]], margin=margin)
    return det


def feature_contributions(z, start, end):
    """Which input features drove a detection window? Mean z^2 per feature."""
    return (z.loc[start:end] ** 2).mean().sort_values(ascending=False)


# ============================================================================
# 4. Detections vs observed degradation
# ============================================================================
def degradation_onset(daily_err, ref_mean, window=90, factor=1.15, persist=30,
                      start="2015-01-01"):
    """First day the rolling mean error exceeds factor*ref_mean for `persist`
    consecutive days. daily_err must be 'higher = worse'."""
    r = daily_err.rolling(window).mean().loc[start:]
    flag = (r > factor * ref_mean).astype(int)
    run = flag.groupby((flag != flag.shift()).cumsum()).cumsum()
    hit = run[run >= persist]
    return None if hit.empty else hit.index[0] - pd.Timedelta(days=persist - 1)


def compare_detections(alarm_dates, onset):
    """Lead (negative) / lag (positive) in days of each alarm relative to onset."""
    if onset is None:
        return pd.Series({pd.Timestamp(a): np.nan for a in alarm_dates})
    return pd.Series({pd.Timestamp(a): (pd.Timestamp(a) - onset).days for a in alarm_dates})


def yearly_input_vs_error(input_score, daily_err):
    """Table + correlation: does more input shift mean more error?"""
    t = pd.DataFrame({"input_score": input_score.groupby(input_score.index.year).mean(),
                      "error": daily_err.groupby(daily_err.index.year).mean()}).dropna()
    return t, t["input_score"].corr(t["error"], method="spearman")


# ============================================================================
# Self-test on synthetic data:  python drift.py
# ============================================================================
def _synthetic():
    idx = pd.date_range("2010-01-01 00:30", "2021-01-01 00:00", freq="30min")
    sh = idx - pd.Timedelta(minutes=30)
    rng = np.random.default_rng(0)
    hr = (sh.hour + sh.minute / 60).to_numpy()
    doy = sh.dayofyear.to_numpy()
    yrs = (sh.year - 2010).to_numpy() + doy / 365.25
    day_noise = pd.Series(rng.normal(0, 1, len(idx))).groupby(sh.normalize()).transform("first").to_numpy()
    temp = 20 + 7 * np.sin(2 * np.pi * (doy - 80) / 365.25) + 3 * day_noise
    solar = np.clip(900 * np.sin(np.pi * (hr - 6) / 12), 0, None) * (0.7 + 0.1 * day_noise)
    ramp = np.clip(yrs - 6, 0, None)  # rooftop solar grows from 2016
    base = 8000 + 60 * np.abs(temp - 20) ** 1.3 + 1500 * np.exp(-((hr - 18) / 3) ** 2)
    dem = base - ramp * 0.9 * solar + rng.normal(0, 150, len(idx)) \
        + np.where(sh.dayofweek >= 5, -700, 0)
    df = pd.DataFrame(index=idx)
    df["nsw_demand"], df["vic_demand"], df["qld_demand"] = dem, 0.6 * dem, 0.7 * dem
    df["nsw_price"] = 60 + 20 * rng.normal(size=len(idx))
    df["syd_temp"], df["syd_solar"] = temp, solar
    df["syd_cloud"], df["syd_rh"] = 50 + 10 * day_noise, 60 + 5 * day_noise
    return df


if __name__ == "__main__":
    df = _synthetic()
    feats = daily_input_features(df)
    sc, det_in, score = build_input_detector(feats, cooldown=90)
    _, in_alarms = det_in.run(score.loc["2015-01-01":])
    print("input-detector alarms:", [str(a.date()) for a in in_alarms])

    pred = pd.DataFrame({"y_true": df["nsw_demand"], "y_pred": df["nsw_demand"].shift(48),
                         "c_true": 1, "c_pred": 1}).iloc[48:]
    err = halfhourly = halfhour_errors(pred)
    de = daily_series(err, "ape")
    det_er = build_error_detector(de, cooldown=90)
    _, er_alarms = det_er.run(de.loc["2015-01-01":])
    print("error-detector alarms:", [str(a.date()) for a in er_alarms])
    onset = degradation_onset(de, de.loc["2014"].mean())
    print("degradation onset:", onset, "| lags (days):", compare_detections(er_alarms, onset).tolist())
    t = monthly_intervals(err[err["day"].dt.year == 2018], "ape", n_boot=200)
    print(t[["month", "value", "width_ind", "width_boot", "width_ratio"]].head(3))
    print("diff CI:", month_difference_ci(err, "2015-06", "2019-06", n_boot=300))
