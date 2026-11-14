import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import holidays
except ImportError:
    holidays = None


def get_nsw_holidays():
    """Returns the NSW public holidays calendar from the holidays package if available."""
    if holidays is not None:
        return holidays.country_holidays("AU", subdiv="NSW", years=range(2010, 2028))
    return set()


def day_block_bootstrap(errors_by_day, stat=np.mean, n_boot: int = 1000, seed: int = 0):
    """
    Appendix A authoritative day-block bootstrap implementation.
    errors_by_day: list of 1D arrays, one per day (each containing 48 half-hour errors).
    """
    rng = np.random.default_rng(seed)
    k = len(errors_by_day)
    stats = [
        stat(np.concatenate([errors_by_day[j] for j in rng.integers(0, k, k)]))
        for _ in range(n_boot)
    ]
    return np.percentile(stats, [2.5, 97.5])


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the half-hourly dataset and assigns each timestamp to its operational day D.
    Because timestamps mark the END of each interval, subtracting 30 minutes maps
    (D 00:30 ... D+1 00:00) cleanly onto operational date D and slot 0..47.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    op_dt = df["timestamp"] - pd.Timedelta(minutes=30)
    df["op_date"] = op_dt.dt.floor("D")
    df["op_year"] = op_dt.dt.year
    df["op_month"] = op_dt.dt.to_period("M")
    df["month_num"] = op_dt.dt.month
    df["half_hour_slot"] = op_dt.dt.hour * 2 + (op_dt.dt.minute // 30)
    df["day_of_week"] = op_dt.dt.day_of_week
    df["hour_of_day"] = df["half_hour_slot"] / 2.0

    nsw_holidays = get_nsw_holidays()
    unique_dates = pd.Series(df["op_date"].dt.date.unique())
    hol_map = {d: (d in nsw_holidays) for d in unique_dates}
    df["is_holiday"] = df["op_date"].dt.date.map(hol_map).astype(bool)
    df["is_weekend"] = df["day_of_week"] >= 5
    return df


def print_summary_statistics(df: pd.DataFrame, out_dir: str):
    """Computes and exports comprehensive summary statistics across all 18 variables and stages."""
    cols = [
        "nsw_demand", "vic_demand", "qld_demand",
        "nsw_price", "vic_price", "qld_price",
        "syd_temp", "syd_rh", "syd_cloud", "syd_solar",
        "mel_temp", "mel_rh", "mel_cloud", "mel_solar",
        "bne_temp", "bne_rh", "bne_cloud", "bne_solar",
    ]
    summary = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["skew"] = df[cols].skew()
    summary.to_csv(os.path.join(out_dir, "table1_overall_summary_stats.csv"))

    df["stage"] = np.where(
        df["op_year"] <= 2013,
        "Stage 1 Train (2010-2013)",
        np.where(df["op_year"] == 2014, "Stage 1 Val (2014)", "Stage 2 Sim (2015-2026)"),
    )
    stage_stats = df.groupby("stage")[
        ["nsw_demand", "vic_demand", "qld_demand", "nsw_price", "syd_temp", "syd_solar"]
    ].agg(["mean", "std", "min", "median", "max"])

    # Compute price regime indicators per stage (negative price %, spike > $300/MWh %)
    price_regime = df.groupby("stage").apply(
        lambda g: pd.Series({
            "neg_price_pct": 100.0 * (g["nsw_price"] < 0).mean(),
            "spike_gt_100_pct": 100.0 * (g["nsw_price"] > 100).mean(),
            "spike_gt_300_pct": 100.0 * (g["nsw_price"] > 300).mean(),
        }),
        include_groups=False,
    )
    stage_stats.to_csv(os.path.join(out_dir, "table2_stage_comparison.csv"))
    price_regime.to_csv(os.path.join(out_dir, "table2b_price_regime_by_stage.csv"))

    print("=== Overall Dataset Summary Statistics ===")
    print(summary[["mean", "std", "min", "5%", "50%", "95%", "max", "skew"]].round(2))
    print("\n=== Stage-by-Stage Mean & Std ===")
    print(stage_stats.round(2))
    print("\n=== Stage-by-Stage NSW Price Regime Indicators ===")
    print(price_regime.round(3))


def plot_demand_and_weather_patterns(df: pd.DataFrame, out_dir: str):
    """
    Generates Figure 1 showing:
      1. Yearly evolution of intraday demand profile (rooftop solar 'duck curve' drift)
      2. Non-linear relationship between Sydney temperature, solar radiation, and NSW demand
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    df["era"] = pd.cut(
        df["op_year"],
        bins=[2009, 2013, 2017, 2021, 2026],
        labels=["2010-2013 (Train)", "2014-2017", "2018-2021", "2022-2026"],
    )
    era_profiles = df.groupby(["hour_of_day", "era"], observed=False)["nsw_demand"].mean().unstack()
    for col in era_profiles.columns:
        if era_profiles[col].notna().any():
            axes[0].plot(era_profiles.index, era_profiles[col], linewidth=2.2, label=str(col))

    axes[0].set_title("(a) Structural Shift in NSW Intraday Demand ('Duck Curve')")
    axes[0].set_xlabel("Time of Day (Hours, AEST)")
    axes[0].set_ylabel("Mean Operational Demand (MW)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Period")

    sample_df = df.sample(n=min(15000, len(df)), random_state=42)
    sc = axes[1].scatter(
        sample_df["syd_temp"],
        sample_df["nsw_demand"],
        c=sample_df["syd_solar"],
        cmap="viridis",
        alpha=0.35,
        s=10,
    )
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("Sydney Shortwave Solar Radiation (W/m²)")
    axes[1].set_title("(b) NSW Demand vs. Sydney Temperature & Solar Radiation")
    axes[1].set_xlabel("Sydney Air Temperature at 2 m (°C)")
    axes[1].set_ylabel("NSW Operational Demand (MW)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_demand_drift_and_weather.png"), dpi=200)
    plt.close(fig)


def plot_comprehensive_demand_and_price_patterns(df: pd.DataFrame, out_dir: str):
    """
    Generates Figure 4 showing detailed patterns in demand and prices for Section 2 of the report:
      (a) Seasonal & Day-of-Week NSW Demand Profiles (Summer, Winter, Shoulder, Weekend/Holiday)
      (b) Intraday NSW Spot Price Profile ($/MWh) & Price Spike Rate (> $100/MWh)
      (c) Yearly NSW Spot Price Distribution (Median, 5th-95th Percentile, and Negative Price Share %)
      (d) Cross-State Demand, Price, and Weather Correlation Matrix
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.2))

    # Panel A: Seasonal & Calendar Demand Profiles
    is_non_work = df["is_weekend"] | df["is_holiday"]
    is_summer = df["month_num"].isin([12, 1, 2])
    is_winter = df["month_num"].isin([6, 7, 8])

    cond_list = [
        (~is_non_work) & is_winter,
        (~is_non_work) & is_summer,
        (~is_non_work) & (~is_summer) & (~is_winter),
        is_non_work,
    ]
    label_list = [
        "Winter Workday (Jun–Aug)",
        "Summer Workday (Dec–Feb)",
        "Shoulder Workday (Spring/Autumn)",
        "Weekend / NSW Public Holiday",
    ]
    colors_a = ["navy", "crimson", "seagreen", "darkorange"]
    for mask, lbl, col in zip(cond_list, label_list, colors_a):
        prof = df.loc[mask].groupby("hour_of_day")["nsw_demand"].mean()
        axes[0, 0].plot(prof.index, prof.values, linewidth=2.2, color=col, label=lbl)
    axes[0, 0].set_title("(a) NSW Intraday Demand by Season and Calendar Regime")
    axes[0, 0].set_xlabel("Time of Day (Hours, AEST)")
    axes[0, 0].set_ylabel("Mean Operational Demand (MW)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="lower right", fontsize=9)

    # Panel B: Intraday NSW Price Profile and High-Price Spike Probability
    ax_b1 = axes[0, 1]
    ax_b2 = ax_b1.twinx()
    hh_price_mean = df.groupby("hour_of_day")["nsw_price"].mean()
    hh_price_med = df.groupby("hour_of_day")["nsw_price"].median()
    hh_spike_pct = df.groupby("hour_of_day")["nsw_price"].apply(lambda s: 100.0 * (s > 100.0).mean())

    l1 = ax_b1.plot(hh_price_mean.index, hh_price_mean.values, color="navy", linewidth=2.2, label="Mean Spot Price ($/MWh)")
    l2 = ax_b1.plot(hh_price_med.index, hh_price_med.values, color="steelblue", linestyle="--", linewidth=1.8, label="Median Spot Price ($/MWh)")
    l3 = ax_b2.plot(hh_spike_pct.index, hh_spike_pct.values, color="crimson", linewidth=2.0, linestyle=":", label="Price > $100/MWh (%)")
    ax_b1.set_title("(b) NSW Intraday Spot Price & Evening Spike Probability")
    ax_b1.set_xlabel("Time of Day (Hours, AEST)")
    ax_b1.set_ylabel("NSW Spot Price ($/MWh)", color="navy")
    ax_b2.set_ylabel("Intervals with Price > $100/MWh (%)", color="crimson")
    ax_b1.grid(True, alpha=0.3)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_b1.legend(lines, labels, loc="upper left", fontsize=9)

    # Panel C: Yearly NSW Price Distribution & Negative Price Share (2010-2026)
    ax_c1 = axes[1, 0]
    ax_c2 = ax_c1.twinx()
    yr_grp = df.groupby("op_year")["nsw_price"]
    yr_med = yr_grp.median()
    yr_p05 = yr_grp.quantile(0.05)
    yr_p95 = yr_grp.quantile(0.95)
    yr_neg = yr_grp.apply(lambda s: 100.0 * (s < 0.0).mean())

    ax_c1.plot(yr_med.index, yr_med.values, color="black", marker="o", linewidth=2.0, label="Median Price ($/MWh)")
    ax_c1.fill_between(yr_med.index, yr_p05.values, yr_p95.values, color="cornflowerblue", alpha=0.28, label="5th–95th Percentile Band")
    ax_c2.bar(yr_neg.index, yr_neg.values, color="crimson", alpha=0.45, width=0.55, label="Negative Price Share (%)")
    ax_c1.set_title("(c) Yearly NSW Spot Price Distribution & Negative Prices")
    ax_c1.set_xlabel("Year")
    ax_c1.set_ylabel("NSW Spot Price ($/MWh)")
    ax_c2.set_ylabel("Negative Price Intervals (%)", color="crimson")
    ax_c1.grid(True, alpha=0.3)
    h1, lbl1 = ax_c1.get_legend_handles_labels()
    h2, lbl2 = ax_c2.get_legend_handles_labels()
    ax_c1.legend(h1 + h2, lbl1 + lbl2, loc="upper left", fontsize=9)

    # Panel D: Cross-State Demand, Log-Price & Weather Correlation Heatmap
    corr_df = pd.DataFrame({
        "NSW Dem": df["nsw_demand"],
        "VIC Dem": df["vic_demand"],
        "QLD Dem": df["qld_demand"],
        "NSW LogP": np.sign(df["nsw_price"]) * np.log1p(np.abs(df["nsw_price"])),
        "VIC LogP": np.sign(df["vic_price"]) * np.log1p(np.abs(df["vic_price"])),
        "QLD LogP": np.sign(df["qld_price"]) * np.log1p(np.abs(df["qld_price"])),
        "Syd Temp": df["syd_temp"],
        "Syd Solar": df["syd_solar"],
    })
    corr_mat = corr_df.corr()
    im = axes[1, 1].imshow(corr_mat.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    axes[1, 1].set_xticks(np.arange(len(corr_mat.columns)))
    axes[1, 1].set_yticks(np.arange(len(corr_mat.columns)))
    axes[1, 1].set_xticklabels(corr_mat.columns, rotation=35, ha="right")
    axes[1, 1].set_yticklabels(corr_mat.columns)
    for i in range(len(corr_mat.columns)):
        for j in range(len(corr_mat.columns)):
            val = corr_mat.values[i, j]
            txt_col = "white" if abs(val) > 0.55 else "black"
            axes[1, 1].text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_col, fontsize=8)
    cbar_d = plt.colorbar(im, ax=axes[1, 1])
    cbar_d.set_label("Pearson Correlation (r)")
    axes[1, 1].set_title("(d) Cross-State Demand, Log-Price & Weather Correlations")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_demand_and_price_patterns.png"), dpi=200)
    plt.close(fig)


def run_day_type_clustering_analysis(df: pd.DataFrame, out_dir: str, n_clusters: int = 4, n_pca: int = 5):
    """
    Implements the Day-Type Analysis required by the COMP9417 specification:
      - Represents each operational day by its 48 half-hourly NSW demand values.
      - Normalises each day's profile by its daily mean (captures daily shape independently of overall volume).
      - Fits PCA and K-Means strictly on the training years (2010-2013) and keeps them fixed.
      - Assigns all days (2010-2026) to the learned clusters and tracks distance to the nearest cluster centre
        as an input-based distribution shift indicator.
    """
    pivot = df.pivot(index="op_date", columns="half_hour_slot", values="nsw_demand").dropna()
    raw_profiles = pivot.to_numpy(dtype=np.float64)
    daily_means = raw_profiles.mean(axis=1, keepdims=True)
    norm_profiles = raw_profiles / np.maximum(daily_means, 1.0)

    years = pivot.index.year
    train_mask = (years >= 2010) & (years <= 2013)
    if not np.any(train_mask):
        train_mask = np.ones(len(pivot), dtype=bool)

    pca = PCA(n_components=min(n_pca, norm_profiles.shape[1]), random_state=42)
    Z_train = pca.fit_transform(norm_profiles[train_mask])
    Z_all = pca.transform(norm_profiles)

    sil_scores = {}
    for k in range(2, min(7, len(Z_train))):
        km_tmp = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z_train)
        sil_scores[k] = float(silhouette_score(Z_train, km_tmp.labels_))
    print("\n=== Day-Type Clustering (2010-2013 Train): Silhouette Scores by k ===")
    for k, score in sil_scores.items():
        print(f"  k = {k}: Silhouette Score = {score:.4f} (Explained Variance by {pca.n_components_} PCs = {100*pca.explained_variance_ratio_.sum():.1f}%)")

    kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=42)
    kmeans.fit(Z_train)

    clusters_all = kmeans.predict(Z_all)
    dists_all = kmeans.transform(Z_all).min(axis=1)

    day_df = pd.DataFrame(
        {
            "year": years,
            "cluster": clusters_all,
            "dist_to_center": dists_all,
            "daily_mean_mw": daily_means.squeeze(-1),
        },
        index=pivot.index,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    hours = np.arange(48) / 2.0

    for c in range(n_clusters):
        c_mask = (clusters_all == c) & train_mask
        if np.any(c_mask):
            mean_prof = norm_profiles[c_mask].mean(axis=0)
            axes[0].plot(hours, mean_prof, linewidth=2.2, label=f"Cluster {c + 1} (n={c_mask.sum()})")
    axes[0].set_title("(a) Learned Day-Type Profiles (2010-2013 Reference)")
    axes[0].set_xlabel("Time of Day (Hours, AEST)")
    axes[0].set_ylabel("Normalised Demand (Ratio to Daily Mean)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    yearly_shares = pd.crosstab(day_df["year"], day_df["cluster"], normalize="index") * 100.0
    yearly_shares.plot(kind="bar", stacked=True, ax=axes[1], width=0.8)
    axes[1].set_title("(b) Yearly Share of Day Types (%)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Share of Days (%)")
    axes[1].legend([f"Cluster {c + 1}" for c in yearly_shares.columns], loc="upper right")
    axes[1].grid(True, axis="y", alpha=0.3)

    yearly_dist = day_df.groupby("year")["dist_to_center"].agg(["mean", lambda x: np.percentile(x, 90)])
    yearly_dist.columns = ["Mean Distance", "90th Percentile Distance"]
    axes[2].plot(yearly_dist.index, yearly_dist["Mean Distance"], marker="o", linewidth=2, label="Mean Distance")
    axes[2].plot(yearly_dist.index, yearly_dist["90th Percentile Distance"], marker="s", linestyle="--", label="90th Percentile")
    axes[2].set_title("(c) Distance to Nearest 2010-2013 Cluster Centre")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Euclidean Distance in PCA Space")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_day_type_clustering_and_drift.png"), dpi=200)
    plt.close(fig)


def plot_price_spikes_and_baseline_ci(df: pd.DataFrame, out_dir: str):
    """
    Analyses price direction target dynamics and computes monthly baseline errors
    with both Independent and Day-Block Bootstrap 95% Confidence Intervals (Appendix A).
    """
    n_days = len(df) // 48
    df_clean = df.iloc[: n_days * 48].copy()

    dem_2d = df_clean["nsw_demand"].to_numpy(dtype=np.float64).reshape(n_days, 48)
    price_2d = df_clean["nsw_price"].to_numpy(dtype=np.float64).reshape(n_days, 48)
    p_bar_d = np.roll(price_2d.mean(axis=1), 1)

    ape_2d = 100.0 * np.abs(dem_2d - np.roll(dem_2d, 1, axis=0)) / dem_2d
    c_true_2d = (price_2d > p_bar_d[:, None]).astype(np.float64)

    day_dates = df_clean["op_date"].iloc[::48].to_numpy()[1:]
    day_months = pd.to_datetime(day_dates).to_period("M")
    ape_days = ape_2d[1:]
    class1_rate = 100.0 * c_true_2d[1:].mean(axis=1)

    records = []
    for month in np.unique(day_months):
        m_mask = day_months == month
        m_ape_days = [ape_days[i] for i in np.where(m_mask)[0]]
        m_ape_flat = np.concatenate(m_ape_days)
        n_hh = len(m_ape_flat)

        mape_val = float(np.mean(m_ape_flat))
        s_ape = float(np.std(m_ape_flat, ddof=1)) if n_hh > 1 else 0.0
        ind_half_width = 1.96 * s_ape / np.sqrt(n_hh)
        boot_low, boot_high = day_block_bootstrap(m_ape_days, stat=np.mean, n_boot=1000, seed=42)

        records.append(
            {
                "month": month.to_timestamp(),
                "mape": mape_val,
                "ind_low": mape_val - ind_half_width,
                "ind_high": mape_val + ind_half_width,
                "boot_low": float(boot_low),
                "boot_high": float(boot_high),
                "class1_share": float(np.mean(class1_rate[m_mask])),
            }
        )

    res_df = pd.DataFrame(records)
    res_df.to_csv(os.path.join(out_dir, "table3_monthly_baseline_ci.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    axes[0].plot(res_df["month"], res_df["mape"], color="navy", linewidth=1.5, label="Monthly Baseline MAPE")
    axes[0].fill_between(
        res_df["month"],
        res_df["boot_low"],
        res_df["boot_high"],
        color="steelblue",
        alpha=0.30,
        label="95% CI: Day-Block Bootstrap",
    )
    axes[0].fill_between(
        res_df["month"],
        res_df["ind_low"],
        res_df["ind_high"],
        color="darkorange",
        alpha=0.45,
        label="95% CI: Independent Half-Hours",
    )
    axes[0].set_title("(a) Baseline Monthly Demand MAPE with 95% Confidence Intervals")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Demand MAPE (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(res_df["month"], res_df["class1_share"], color="crimson", linewidth=1.6, label="Monthly Class 1 Share")
    axes[1].axhline(50.0, color="black", linestyle="--", alpha=0.5, label="50% Balanced Line")
    axes[1].set_title("(b) Monthly Share of Positive Price Direction (P_{D+1,t} > Mean(P_D))")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Share of Class 1 Half-Hours (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_monthly_ci_and_price_class_balance.png"), dpi=200)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    default_csv = os.path.join(root, "data", "nem_weather_halfhourly_2010_2026.csv")
    default_out = os.path.join(here, "analysis_outputs")

    parser = argparse.ArgumentParser(description="COMP9417 Exploratory Data & Drift Analysis")
    parser.add_argument("csv_path", nargs="?", default=default_csv, help="Path to nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--out_dir", type=str, default=default_out, help="Output folder for plots and tables")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_prepare_data(args.csv_path)
    print_summary_statistics(df, args.out_dir)
    plot_demand_and_weather_patterns(df, args.out_dir)
    plot_comprehensive_demand_and_price_patterns(df, args.out_dir)
    run_day_type_clustering_analysis(df, args.out_dir)
    plot_price_spikes_and_baseline_ci(df, args.out_dir)
    print(f"\nAll exploratory plots and summary tables saved to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
