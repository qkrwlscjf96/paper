from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_boxplots_by_date(
    df: pd.DataFrame,
    check_cols: list,
    target_col: str = "TAG",
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """EDA 결과물을 저장하는 시각화 함수."""
    plt.style.use("ggplot")

    output_path = Path(output_dir) if output_dir is not None else Path("result") / "eda"
    output_path.mkdir(parents=True, exist_ok=True)

    valid_cols = [col for col in check_cols if pd.api.types.is_numeric_dtype(df[col])]

    overview_df = (
        df.assign(IS_NG=df[target_col].astype(int))
        .groupby("DATE")
        .agg(
            total_count=(target_col, "size"),
            ng_count=(target_col, "sum"),
        )
        .reset_index()
    )
    overview_df["ng_rate"] = overview_df["ng_count"] / overview_df["total_count"]

    feature_summary = []
    for col in valid_cols:
        grouped = df.groupby(target_col)[col].mean()
        ok_mean = grouped.get(0, pd.NA)
        ng_mean = grouped.get(1, pd.NA)
        feature_summary.append(
            {
                "FEATURE": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_ratio": float(df[col].isna().mean()),
                "mean_all": float(df[col].mean()),
                "std_all": float(df[col].std()),
                "ok_mean": ok_mean,
                "ng_mean": ng_mean,
                "mean_gap_ng_minus_ok": (
                    float(ng_mean - ok_mean)
                    if pd.notna(ok_mean) and pd.notna(ng_mean)
                    else pd.NA
                ),
            }
        )

    feature_summary_df = pd.DataFrame(feature_summary).sort_values(
        "mean_gap_ng_minus_ok",
        key=lambda series: series.abs(),
        ascending=False,
        na_position="last",
    )

    overview_csv = output_path / "date_overview.csv"
    feature_csv = output_path / "feature_summary.csv"
    overview_df.to_csv(overview_csv, index=False)
    feature_summary_df.to_csv(feature_csv, index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(overview_df["DATE"], overview_df["ng_rate"], marker="o")
    plt.title("NG Rate by Date")
    plt.xlabel("DATE")
    plt.ylabel("NG Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ng_rate_path = output_path / "ng_rate_by_date.png"
    plt.savefig(ng_rate_path, dpi=180, bbox_inches="tight")
    plt.close()

    if valid_cols:
        scatter_rows = len(valid_cols)
        fig, axes = plt.subplots(scatter_rows, 1, figsize=(16, 4 * scatter_rows), squeeze=False)
        tag_values = sorted(df[target_col].dropna().unique())
        colors = {0: "#1f77b4", 1: "#d62728"}

        for ax, col in zip(axes.flatten(), valid_cols):
            for tag_value in tag_values:
                subset = df[df[target_col] == tag_value]
                ax.scatter(
                    subset["DATE"],
                    subset[col],
                    alpha=0.65,
                    s=18,
                    color=colors.get(tag_value, "#7f7f7f"),
                    label=f"{target_col}={tag_value}",
                )
            ax.set_title(f"{col} Scatter by Date")
            ax.set_xlabel("DATE")
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title=target_col, loc="upper right")

        fig.tight_layout()
        scatter_path = output_path / "all_features_scatter.png"
        fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        daily_mean_df = df.groupby("DATE")[valid_cols].mean().reset_index()
        fig, axes = plt.subplots(scatter_rows, 1, figsize=(16, 4 * scatter_rows), squeeze=False)

        for ax, col in zip(axes.flatten(), valid_cols):
            ax.plot(daily_mean_df["DATE"], daily_mean_df[col], color="#2ca02c", linewidth=1.8)
            ax.set_title(f"{col} Time Series by Date")
            ax.set_xlabel("DATE")
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        timeseries_path = output_path / "all_features_timeseries.png"
        fig.savefig(timeseries_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        melted_df = df.melt(
            id_vars=["DATE", target_col],
            value_vars=valid_cols,
            var_name="FEATURE",
            value_name="VALUE",
        )
        fig, axes = plt.subplots(1, len(valid_cols), figsize=(max(14, len(valid_cols) * 3.2), 6), squeeze=False)

        for ax, col in zip(axes.flatten(), valid_cols):
            box_data = []
            labels = []
            for tag_value in tag_values:
                values = melted_df[
                    (melted_df["FEATURE"] == col) &
                    (melted_df[target_col] == tag_value)
                ]["VALUE"].dropna()
                if not values.empty:
                    box_data.append(values)
                    labels.append(f"{tag_value} (n={len(values)})")

            if box_data:
                ax.boxplot(box_data, labels=labels, showfliers=False)
            ax.set_title(col)
            ax.set_xlabel(target_col)
            ax.set_ylabel("VALUE")

        fig.suptitle("All Feature Distribution by TAG")
        fig.tight_layout()
        boxplot_path = output_path / "all_features_boxplot.png"
        fig.savefig(boxplot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        scatter_path = None
        timeseries_path = None
        boxplot_path = None

    print(
        "EDA 결과 저장 완료: "
        f"output_dir={output_path}, "
        f"feature_count={len(valid_cols)}"
    )

    return {
        "output_dir": str(output_path),
        "date_overview_csv": str(overview_csv),
        "feature_summary_csv": str(feature_csv),
        "ng_rate_plot": str(ng_rate_path),
        "all_features_scatter": str(scatter_path) if scatter_path else "",
        "all_features_timeseries": str(timeseries_path) if timeseries_path else "",
        "all_features_boxplot": str(boxplot_path) if boxplot_path else "",
    }
