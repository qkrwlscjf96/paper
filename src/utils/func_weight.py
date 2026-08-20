import pandas as pd


def build_statistical_sample_weights(
    df: pd.DataFrame,
    static_idx_result: pd.DataFrame | None,
    static_date_result: pd.DataFrame | None,
    weight_mul: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply one row weight to the union of statistical index/date matches."""
    if weight_mul < 1:
        raise ValueError("weight_mul must be at least 1")

    index_mask = pd.Series(False, index=df.index)
    if static_idx_result is not None and not static_idx_result.empty:
        index_mask = df.index.to_series().isin(
            static_idx_result["INDEX"].dropna().unique()
        )

    date_mask = pd.Series(False, index=df.index)
    if (
        static_date_result is not None
        and not static_date_result.empty
        and "DATE" in static_date_result.columns
    ):
        date_mask = df["DATE"].isin(static_date_result["DATE"].dropna().unique())

    weighted_mask = index_mask | date_mask
    sample_weights = pd.Series(1.0, index=df.index, name="sample_weight")
    sample_weights.loc[weighted_mask] = float(weight_mul)
    diagnostics = pd.DataFrame(
        {
            "INDEX": df.index,
            "flag_static_index": index_mask.astype(int),
            "flag_static_date": date_mask.astype(int),
            "sample_weight": sample_weights,
        }
    )
    return sample_weights, diagnostics
