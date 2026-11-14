"""Feature engineering, Day-Type clustering transformer, and hybrid feature selector
for NSW Day-Ahead Electricity Demand Forecasting (COMP9417).

All feature extraction strictly respects the chronological boundary at the end of Day D:
only historical market/weather observations up to Day D (interval D+1 00:00),
forecast weather for Day D+1, and calendar/holiday information are used.
"""
from typing import Dict, List, Optional, Tuple
import warnings
import holidays
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

N_PERIODS = 48
WINDOW_DAYS = 7
MARKET_COLS = [
    "nsw_demand", "nsw_price",
    "qld_demand", "qld_price",
    "vic_demand", "vic_price",
]
WEATHER_COLS = [
    f"{city}_{var}"
    for city in ("syd", "mel", "bne")
    for var in ("temp", "rh", "cloud", "solar")
]


def get_nsw_holidays(years=range(2010, 2028)):
    """Return NSW public holidays calendar from the holidays package."""
    return holidays.country_holidays("AU", subdiv="NSW", years=years)


def market_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each half-hour timestamp (end of interval) to its operational day D."""
    return (index - pd.Timedelta("30min")).floor("D")


def mape_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100.0)


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MW)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


class DayTypeTransformer:
    """Unsupervised Day-Type feature extractor using PCA + K-Means on mean-normalised
    daily 48-half-hour NSW demand profiles.

    Fitted strictly on training days (e.g., 2010-2013 during Stage 1 selection,
    and 2010-2014 when refitting the frozen Stage 1 model). When predicting Day D+1,
    only historical days (Day D and Day D-6) are transformed, preventing any leakage.
    """

    def __init__(self, n_components: int = 5, n_clusters: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.kmeans: Optional[KMeans] = None

    @staticmethod
    def normalise_profiles(raw_profiles_2d: np.ndarray) -> np.ndarray:
        """Divide each day's 48-half-hour profile by its daily mean demand."""
        daily_means = np.mean(raw_profiles_2d, axis=1, keepdims=True)
        return raw_profiles_2d / np.maximum(daily_means, 1.0)

    def fit(self, raw_profiles_2d: np.ndarray) -> "DayTypeTransformer":
        norm_prof = self.normalise_profiles(raw_profiles_2d)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        z_train = self.pca.fit_transform(norm_prof)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state,
        )
        self.kmeans.fit(z_train)
        return self

    def transform(self, raw_profiles_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cluster_labels, min_euclidean_distance_to_center) for each day."""
        norm_prof = self.normalise_profiles(raw_profiles_2d)
        z_all = self.pca.transform(norm_prof)
        clusters = self.kmeans.predict(z_all)
        dists = self.kmeans.transform(z_all).min(axis=1)
        return clusters, dists


def build_candidate_features(
    df: pd.DataFrame,
    day_type_transformer: Optional[DayTypeTransformer] = None,
    fit_day_type_max_year: int = 2013,
    include_flat_daily: bool = False,
) -> Dict[str, object]:
    """Build the 3D slot-aligned feature tensor X_3d of shape (n_samples, 48, n_features)
    and target demand matrix Y_2d of shape (n_samples, 48).

    Parameters
    ----------
    df : pd.DataFrame
        Half-hourly dataset indexed by timestamp, sorted chronologically.
    day_type_transformer : DayTypeTransformer, optional
        Pre-fitted DayTypeTransformer. If None, a new transformer is fitted strictly
        on days with year <= fit_day_type_max_year.
    fit_day_type_max_year : int
        Cutoff year (inclusive) for fitting DayTypeTransformer when none is passed.
    include_flat_daily : bool
        If True, also builds X_flat (n_samples, 403) representing the unaligned daily
        multi-output feature vector for baseline architecture comparison.
    """
    days = market_day(df.index)
    counts = pd.Series(1, index=df.index).groupby(days).sum()
    valid_days = counts[counts == N_PERIODS].index

    if len(valid_days) == len(df) // N_PERIODS:
        df_clean = df
        unique_days = pd.DatetimeIndex(valid_days)
    else:
        mask = days.isin(valid_days)
        df_clean = df.loc[mask]
        unique_days = pd.DatetimeIndex(sorted(valid_days))

    n_days = len(unique_days)

    def col_2d(col_name: str) -> np.ndarray:
        if col_name in df_clean.columns:
            return df_clean[col_name].to_numpy(dtype=np.float64).reshape(n_days, N_PERIODS)
        return np.zeros((n_days, N_PERIODS), dtype=np.float64)

    nsw_dem = col_2d("nsw_demand")
    vic_dem = col_2d("vic_demand")
    qld_dem = col_2d("qld_demand")
    nsw_prc = col_2d("nsw_price")

    syd_temp = col_2d("syd_temp")
    syd_rh = col_2d("syd_rh")
    syd_cloud = col_2d("syd_cloud")
    syd_solar = col_2d("syd_solar")

    mel_temp = col_2d("mel_temp")
    bne_temp = col_2d("bne_temp")

    nsw_holidays = get_nsw_holidays(range(unique_days[0].year, unique_days[-1].year + 2))
    is_hol = np.array([1.0 if d.date() in nsw_holidays else 0.0 for d in unique_days], dtype=np.float64)
    dow = unique_days.day_of_week.to_numpy()
    is_wknd = (dow >= 5).astype(np.float64)
    is_non_work = np.maximum(is_wknd, is_hol)
    doy = unique_days.day_of_year.to_numpy(dtype=np.float64)

    # Fit or apply DayTypeTransformer on historical demand profiles
    if day_type_transformer is None:
        day_type_transformer = DayTypeTransformer(n_components=5, n_clusters=4, random_state=42)
        fit_mask = unique_days.year <= fit_day_type_max_year
        if not np.any(fit_mask):
            fit_mask = np.ones(n_days, dtype=bool)
        day_type_transformer.fit(nsw_dem[fit_mask])

    # Transform all days up to Day D (note: target day D+1 is at index i, we only ever index i-1 and i-7!)
    # For single-day live prediction, the last row of nsw_dem is a dummy target placeholder, so avoid PCA errors on zeros
    safe_nsw_dem = nsw_dem.copy()
    zero_rows = safe_nsw_dem.mean(axis=1) <= 0
    if np.any(zero_rows):
        safe_nsw_dem[zero_rows] = safe_nsw_dem[0]
    clusters_all, dists_all = day_type_transformer.transform(safe_nsw_dem)

    # Filter indices i >= 7 where 7 consecutive days of history exist
    all_idx = np.arange(WINDOW_DAYS, n_days)
    day_span = (unique_days[all_idx] - unique_days[all_idx - WINDOW_DAYS]).days == WINDOW_DAYS
    idx = all_idx[day_span]
    target_dates = unique_days[idx]
    n_samples = len(idx)

    feats: Dict[str, np.ndarray] = {}
    groups: Dict[str, List[str]] = {}

    def add_feat(name: str, arr: np.ndarray, group: str) -> None:
        if arr.ndim == 1:
            arr_2d = np.broadcast_to(arr[:, None], (n_samples, N_PERIODS))
        else:
            arr_2d = arr
        feats[name] = arr_2d
        groups.setdefault(group, []).append(name)

    # 1. Core NSW Demand Lags (slot-aligned & adjacent half-hours)
    dem_d = nsw_dem[idx - 1]       # Day D (yesterday)
    dem_d1 = nsw_dem[idx - 2]      # Day D-1 (2 days ago)
    dem_d2 = nsw_dem[idx - 3]      # Day D-2 (3 days ago)
    dem_d6 = nsw_dem[idx - 7]      # Day D-6 (same day last week)

    add_feat("dem_d_t", dem_d, "nsw_lags")
    add_feat("dem_d1_t", dem_d1, "nsw_lags")
    add_feat("dem_d2_t", dem_d2, "nsw_lags")
    add_feat("dem_d6_t", dem_d6, "nsw_lags")

    dem_d_prev_hh = np.roll(dem_d, 1, axis=1)
    dem_d_prev_hh[:, 0] = nsw_dem[idx - 2, -1]
    dem_d_next_hh = np.roll(dem_d, -1, axis=1)
    dem_d_next_hh[:, -1] = dem_d[:, -1]
    add_feat("dem_d_prev_hh", dem_d_prev_hh, "nsw_lags")
    add_feat("dem_d_next_hh", dem_d_next_hh, "nsw_lags")

    # Last interval of Day D (D+1 00:00) and evening peak (18:00)
    add_feat("dem_d_last_hh", dem_d[:, -1], "nsw_lags")
    add_feat("dem_d_evening_hh", dem_d[:, 36], "nsw_lags")

    # 2. Rolling & Daily Summary Demand Statistics
    roll7_slot = sum(nsw_dem[idx - k] for k in range(1, 8)) / 7.0
    roll3_slot = sum(nsw_dem[idx - k] for k in range(1, 4)) / 3.0
    add_feat("dem_roll7_t", roll7_slot, "rolling_stats")
    add_feat("dem_roll3_t", roll3_slot, "rolling_stats")
    add_feat("dem_d_mean", dem_d.mean(axis=1), "rolling_stats")
    add_feat("dem_d_max", dem_d.max(axis=1), "rolling_stats")
    add_feat("dem_d_min", dem_d.min(axis=1), "rolling_stats")
    add_feat("dem_d_std", dem_d.std(axis=1), "rolling_stats")
    add_feat("dem_d6_mean", dem_d6.mean(axis=1), "rolling_stats")
    add_feat("dem_roll7_mean", roll7_slot.mean(axis=1), "rolling_stats")
    add_feat("dem_diff_d_d6_t", dem_d - dem_d6, "rolling_stats")

    # 3. Sydney Weather for Forecast Day D+1 (known reanalysis forecast)
    t_next = syd_temp[idx]
    t_d = syd_temp[idx - 1]
    t_d6 = syd_temp[idx - 7]
    rh_next = syd_rh[idx]
    rh_d = syd_rh[idx - 1]
    cl_next = syd_cloud[idx]
    sol_next = syd_solar[idx]
    sol_d = syd_solar[idx - 1]

    cdd_next = np.maximum(0.0, t_next - 22.0)
    hdd_next = np.maximum(0.0, 18.0 - t_next)
    cdd_sq_next = cdd_next ** 2
    hdd_sq_next = hdd_next ** 2

    cdd_d = np.maximum(0.0, t_d - 22.0)
    hdd_d = np.maximum(0.0, 18.0 - t_d)
    cdd_d6 = np.maximum(0.0, t_d6 - 22.0)
    hdd_d6 = np.maximum(0.0, 18.0 - t_d6)

    add_feat("syd_temp_t", t_next, "syd_weather")
    add_feat("syd_temp_sq_t", t_next ** 2, "syd_weather")
    add_feat("syd_cdd_t", cdd_next, "syd_weather")
    add_feat("syd_hdd_t", hdd_next, "syd_weather")
    add_feat("syd_cdd_sq_t", cdd_sq_next, "syd_weather")
    add_feat("syd_hdd_sq_t", hdd_sq_next, "syd_weather")
    add_feat("syd_rh_t", rh_next, "syd_weather")
    add_feat("syd_cloud_t", cl_next, "syd_weather")
    add_feat("syd_solar_t", sol_next, "syd_weather")
    add_feat("syd_temp_x_rh_t", cdd_next * (rh_next / 100.0), "syd_weather")
    add_feat("syd_temp_max", t_next.max(axis=1), "syd_weather")
    add_feat("syd_temp_min", t_next.min(axis=1), "syd_weather")
    add_feat("syd_temp_mean", t_next.mean(axis=1), "syd_weather")
    add_feat("syd_solar_mean", sol_next.mean(axis=1), "syd_weather")

    # 4. Weather Deltas (Day D+1 minus Day D and Day D-6)
    add_feat("delta_temp_d_t", t_next - t_d, "weather_deltas")
    add_feat("delta_cdd_d_t", cdd_next - cdd_d, "weather_deltas")
    add_feat("delta_hdd_d_t", hdd_next - hdd_d, "weather_deltas")
    add_feat("delta_solar_d_t", sol_next - sol_d, "weather_deltas")
    add_feat("delta_rh_d_t", rh_next - rh_d, "weather_deltas")
    add_feat("delta_temp_d6_t", t_next - t_d6, "weather_deltas")
    add_feat("delta_cdd_d6_t", cdd_next - cdd_d6, "weather_deltas")
    add_feat("delta_hdd_d6_t", hdd_next - hdd_d6, "weather_deltas")
    add_feat("syd_temp_d_max", t_d.max(axis=1), "weather_deltas")

    # 5. Calendar, Harmonics & Regime Indicators
    for d_i in range(7):
        add_feat(f"dow_{d_i}", (dow[idx] == d_i).astype(np.float64), "calendar")
    add_feat("is_weekend", is_wknd[idx], "calendar")
    add_feat("is_holiday", is_hol[idx], "calendar")
    add_feat("is_non_work", is_non_work[idx], "calendar")
    add_feat("is_non_work_d", is_non_work[idx - 1], "calendar")
    add_feat("is_non_work_d6", is_non_work[idx - 7], "calendar")
    add_feat("sin_doy_1", np.sin(2.0 * np.pi * doy[idx] / 365.25), "calendar")
    add_feat("cos_doy_1", np.cos(2.0 * np.pi * doy[idx] / 365.25), "calendar")
    add_feat("sin_doy_2", np.sin(4.0 * np.pi * doy[idx] / 365.25), "calendar")
    add_feat("cos_doy_2", np.cos(4.0 * np.pi * doy[idx] / 365.25), "calendar")

    # 6. Regime-Transition Lag Interactions (critical for Mondays, Saturdays, Holidays)
    same_regime_d = (is_non_work[idx] == is_non_work[idx - 1]).astype(np.float64)[:, None]
    diff_regime_d = 1.0 - same_regime_d
    add_feat("dem_d_x_same_regime", dem_d * same_regime_d, "regime_interactions")
    add_feat("dem_d_x_diff_regime", dem_d * diff_regime_d, "regime_interactions")
    add_feat("dem_d6_x_same_regime", dem_d6 * same_regime_d, "regime_interactions")
    add_feat("dem_d6_x_diff_regime", dem_d6 * diff_regime_d, "regime_interactions")
    add_feat("dem_d_x_hol", dem_d * is_hol[idx][:, None], "regime_interactions")
    add_feat("dem_d6_x_hol", dem_d6 * is_hol[idx][:, None], "regime_interactions")

    # 7. Interstate Variables (VIC & QLD Demand, Weather, and NSW Lagged Spot Price)
    add_feat("vic_dem_d_t", vic_dem[idx - 1], "interstate")
    add_feat("qld_dem_d_t", qld_dem[idx - 1], "interstate")
    add_feat("vic_dem_d6_t", vic_dem[idx - 7], "interstate")
    add_feat("qld_dem_d6_t", qld_dem[idx - 7], "interstate")
    add_feat("mel_temp_t", mel_temp[idx], "interstate")
    add_feat("bne_temp_t", bne_temp[idx], "interstate")
    add_feat(
        "nsw_log_price_d_t",
        np.sign(nsw_prc[idx - 1]) * np.log1p(np.abs(nsw_prc[idx - 1])),
        "interstate",
    )
    prc_d_mean = nsw_prc[idx - 1].mean(axis=1)
    add_feat(
        "nsw_price_d_mean",
        np.sign(prc_d_mean) * np.log1p(np.abs(prc_d_mean)),
        "interstate",
    )

    # 8. Learned Historical Day-Type Cluster Features (Day D and Day D-6)
    n_clusters = day_type_transformer.n_clusters
    for c in range(n_clusters):
        add_feat(
            f"day_d_cluster_{c}",
            (clusters_all[idx - 1] == c).astype(np.float64),
            "day_type",
        )
        add_feat(
            f"day_d6_cluster_{c}",
            (clusters_all[idx - 7] == c).astype(np.float64),
            "day_type",
        )
    add_feat("day_d_cluster_dist", dists_all[idx - 1], "day_type")

    feature_names = list(feats.keys())
    X_3d = np.stack([feats[k] for k in feature_names], axis=-1)
    Y_2d = nsw_dem[idx]

    result = {
        "X_3d": X_3d,
        "Y_2d": Y_2d,
        "target_dates": target_dates,
        "feature_names": feature_names,
        "feature_groups": groups,
        "day_type_transformer": day_type_transformer,
    }

    if include_flat_daily:
        X_flat_list = []
        for i in idx:
            d_dem = nsw_dem[i - 1]
            d1_dem = nsw_dem[i - 2]
            d6_dem = nsw_dem[i - 7]
            lag_f = np.concatenate([
                d_dem, d6_dem,
                [d_dem.mean(), d_dem.min(), d_dem.max(), d1_dem.mean(), d6_dem.mean()],
            ])
            t_temp = syd_temp[i]
            t_sol = syd_solar[i]
            t_cld = syd_cloud[i]
            cdd = np.maximum(0.0, t_temp - 22.0)
            hdd = np.maximum(0.0, 18.0 - t_temp)
            wea_f = np.concatenate([
                t_temp, t_temp ** 2, cdd, hdd, t_sol, t_cld,
                [t_temp.max(), t_temp.min(), t_sol.mean()],
            ])
            dow_ohe = np.zeros(7, dtype=np.float64)
            dow_ohe[dow[i]] = 1.0
            cal_f = np.concatenate([
                dow_ohe,
                [
                    is_wknd[i],
                    is_hol[i],
                    np.sin(2.0 * np.pi * doy[i] / 365.25),
                    np.cos(2.0 * np.pi * doy[i] / 365.25),
                ],
            ])
            X_flat_list.append(np.concatenate([lag_f, wea_f, cal_f]))
        result["X_flat"] = np.asarray(X_flat_list, dtype=np.float64)

    return result


def build_single_day_features(
    history: pd.DataFrame,
    weather_next: pd.DataFrame,
    day_type_transformer: DayTypeTransformer,
) -> np.ndarray:
    """Build the (48, n_features) slot-aligned feature matrix for a single target day D+1
    using only the last 7 complete days of history and weather_next.
    """
    recent_history = history.iloc[-N_PERIODS * WINDOW_DAYS:].copy()
    target_placeholder = pd.DataFrame(0.0, index=weather_next.index, columns=history.columns)
    for col in weather_next.columns:
        if col in target_placeholder.columns:
            target_placeholder[col] = weather_next[col].to_numpy(dtype=np.float64)

    combined = pd.concat([recent_history, target_placeholder], axis=0)
    bundle = build_candidate_features(
        combined,
        day_type_transformer=day_type_transformer,
        include_flat_daily=False,
    )
    return bundle["X_3d"][-1]


def evaluate_subset_ridge(
    X_3d: np.ndarray,
    Y_2d: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_indices: List[int],
    alpha: float = 5.0,
) -> Tuple[float, float]:
    """Fit per-slot Ridge regression on train_mask and return (MAPE, MAE) on val_mask."""
    X_sub = X_3d[:, :, feature_indices]
    X_tr, Y_tr = X_sub[train_mask], Y_2d[train_mask]
    X_va, Y_va = X_sub[val_mask], Y_2d[val_mask]
    preds = np.zeros_like(Y_va)
    for t in range(N_PERIODS):
        sc = StandardScaler()
        xt_tr = sc.fit_transform(X_tr[:, t, :])
        xt_va = sc.transform(X_va[:, t, :])
        model = Ridge(alpha=alpha)
        model.fit(xt_tr, Y_tr[:, t])
        preds[:, t] = model.predict(xt_va)
    return mape_metric(Y_va, preds), mae_metric(Y_va, preds)


class FeatureSelector:
    """Two-stage feature selector for Day-Ahead NSW Electricity Demand Forecasting:

    1. Domain Feature Group Ablation (progressive forward addition + leave-one-group-out)
       evaluated on the 2014 validation set.
    2. Hybrid L1 (Lasso) + Residual/Level F-Statistic ranking fitted strictly on the
       2010-2013 training set, followed by validation-guided Top-K cutoff selection on 2014.
    """

    def __init__(
        self,
        lasso_alpha: float = 0.30,
        l1_weight: float = 0.85,
        f_weight: float = 0.15,
        eval_ridge_alpha: float = 5.0,
        random_state: int = 42,
    ):
        self.lasso_alpha = lasso_alpha
        self.l1_weight = l1_weight
        self.f_weight = f_weight
        self.eval_ridge_alpha = eval_ridge_alpha
        self.random_state = random_state

        self.ablation_df_: Optional[pd.DataFrame] = None
        self.ranking_df_: Optional[pd.DataFrame] = None
        self.k_curve_df_: Optional[pd.DataFrame] = None
        self.best_k_: Optional[int] = None
        self.selected_indices_: Optional[List[int]] = None
        self.selected_feature_names_: Optional[List[str]] = None

    def fit_and_select(
        self,
        X_3d: np.ndarray,
        Y_2d: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        feature_names: List[str],
        feature_groups: Dict[str, List[str]],
        k_candidates: Optional[List[int]] = None,
    ) -> "FeatureSelector":
        """Run group ablation, compute training L1 + Residual-F ranking, and select best K on 2014."""
        warnings.filterwarnings("ignore")

        # Stage A: Feature Group Ablation (Forward Progressive + Leave-One-Group-Out)
        ablation_rows = []
        cumulative_cols: List[str] = []
        for g_name, g_cols in feature_groups.items():
            cumulative_cols = cumulative_cols + list(g_cols)
            col_idx = [feature_names.index(c) for c in cumulative_cols]
            mape_v, mae_v = evaluate_subset_ridge(
                X_3d, Y_2d, train_mask, val_mask, col_idx, alpha=self.eval_ridge_alpha
            )
            ablation_rows.append({
                "mode": "progressive_add",
                "group": g_name,
                "n_features": len(col_idx),
                "val_mape": mape_v,
                "val_mae": mae_v,
            })

        full_mape, full_mae = ablation_rows[-1]["val_mape"], ablation_rows[-1]["val_mae"]
        for g_name, g_cols in feature_groups.items():
            if g_name == "nsw_lags":
                continue
            rem_cols = [c for c in feature_names if c not in g_cols]
            col_idx = [feature_names.index(c) for c in rem_cols]
            mape_v, mae_v = evaluate_subset_ridge(
                X_3d, Y_2d, train_mask, val_mask, col_idx, alpha=self.eval_ridge_alpha
            )
            ablation_rows.append({
                "mode": "leave_one_out",
                "group": f"without_{g_name}",
                "n_features": len(col_idx),
                "val_mape": mape_v,
                "val_mae": mae_v,
                "delta_mape_vs_full": mape_v - full_mape,
            })
        self.ablation_df_ = pd.DataFrame(ablation_rows)

        # Stage B: Hybrid L1 (Lasso) + Residual F-Statistic Ranking on 2010-2013 Train
        X_tr, Y_tr = X_3d[train_mask], Y_2d[train_mask]
        dem_d_idx = feature_names.index("dem_d_t")
        Y_res_tr = Y_tr - X_tr[:, :, dem_d_idx]

        n_feats = len(feature_names)
        l1_imp = np.zeros(n_feats, dtype=np.float64)
        f_lvl_imp = np.zeros(n_feats, dtype=np.float64)
        f_res_imp = np.zeros(n_feats, dtype=np.float64)

        for t in range(N_PERIODS):
            sc = StandardScaler()
            xt = sc.fit_transform(X_tr[:, t, :])
            yt = Y_tr[:, t]

            lasso = Lasso(alpha=self.lasso_alpha, max_iter=3000, random_state=self.random_state)
            lasso.fit(xt, yt)
            l1_imp += np.abs(lasso.coef_) / float(N_PERIODS)

            f_lvl, _ = f_regression(xt, yt)
            f_res, _ = f_regression(xt, Y_res_tr[:, t])
            f_lvl = np.nan_to_num(f_lvl, nan=0.0, posinf=0.0, neginf=0.0)
            f_res = np.nan_to_num(f_res, nan=0.0, posinf=0.0, neginf=0.0)
            f_lvl_imp += np.log1p(f_lvl) / float(N_PERIODS)
            f_res_imp += np.log1p(f_res) / float(N_PERIODS)

        l1_norm = l1_imp / (l1_imp.max() + 1e-12)
        f_res_norm = f_res_imp / (f_res_imp.max() + 1e-12)
        composite_score = self.l1_weight * l1_norm + self.f_weight * f_res_norm
        ranked_indices = np.argsort(composite_score)[::-1]

        feat_to_group = {
            f: g for g, cols in feature_groups.items() for f in cols
        }

        # Evaluate Top-K subsets on 2014 Validation
        if k_candidates is None:
            k_candidates = [25, 35, 45, 50, 55, 60, 62, 65, 68, 70, 72, 75, n_feats]
        k_candidates = sorted(set(min(k, n_feats) for k in k_candidates))

        k_rows = []
        for k in k_candidates:
            sub_idx = ranked_indices[:k].tolist()
            mape_v, mae_v = evaluate_subset_ridge(
                X_3d, Y_2d, train_mask, val_mask, sub_idx, alpha=self.eval_ridge_alpha
            )
            k_rows.append({
                "k": int(k),
                "val_mape": mape_v,
                "val_mae": mae_v,
            })
        self.k_curve_df_ = pd.DataFrame(k_rows)

        best_row = self.k_curve_df_.loc[self.k_curve_df_["val_mape"].idxmin()]
        self.best_k_ = int(best_row["k"])
        self.selected_indices_ = ranked_indices[: self.best_k_].tolist()
        self.selected_feature_names_ = [feature_names[i] for i in self.selected_indices_]
        selected_set = set(self.selected_indices_)

        ranking_rows = []
        for rank_pos, idx_f in enumerate(ranked_indices, start=1):
            fname = feature_names[idx_f]
            ranking_rows.append({
                "rank": rank_pos,
                "feature_index": int(idx_f),
                "feature_name": fname,
                "group": feat_to_group.get(fname, "other"),
                "l1_importance_mw": float(l1_imp[idx_f]),
                "log_f_level": float(f_lvl_imp[idx_f]),
                "log_f_residual": float(f_res_imp[idx_f]),
                "composite_score": float(composite_score[idx_f]),
                "selected": idx_f in selected_set,
            })
        self.ranking_df_ = pd.DataFrame(ranking_rows)
        return self
