import numpy as np
import pandas as pd


def _safe_pearson_corr(left: pd.Series, right: pd.Series) -> float:
    """Return Pearson correlation without warnings for constant/invalid series."""
    paired = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 2:
        return np.nan

    left_values = paired.iloc[:, 0].to_numpy(dtype=float)
    right_values = paired.iloc[:, 1].to_numpy(dtype=float)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return np.nan

    correlation = np.dot(left_centered, right_centered) / denominator
    return float(np.clip(correlation, -1.0, 1.0))


def _ensure_date_series(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    return result


def _normalize_index_result(result: pd.DataFrame) -> pd.DataFrame:
    normalized = result.copy()
    if "INDEX" not in normalized.columns:
        normalized["INDEX"] = normalized.index
    if "DATE" in normalized.columns:
        normalized["DATE"] = pd.to_datetime(normalized["DATE"])

    ordered_cols = ["INDEX"]
    if "DATE" in normalized.columns:
        ordered_cols.append("DATE")
    ordered_cols.extend(col for col in normalized.columns if col not in ordered_cols)
    return normalized[ordered_cols].reset_index(drop=True)


def _empty_date_feature_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["DATE", "FEATURE"])


def _normalize_date_feature_result(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return _empty_date_feature_result()

    normalized = result.copy()
    normalized["DATE"] = pd.to_datetime(normalized["DATE"])
    ordered_cols = ["DATE", "FEATURE"] + [
        col for col in normalized.columns if col not in ["DATE", "FEATURE"]
    ]
    return normalized[ordered_cols].reset_index(drop=True)


class FuncStaticV1:
    """기존 통계 분석 로직 모음."""

    @staticmethod
    def iqr_remover(
        full_df: pd.DataFrame,
        target_col: str,
        check_cols: list,
        directional_corr_threshold: float = 0.0,
        return_details: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """IQR 기반 이상치 탐지 함수."""

        iqr_bounds = {}

        for col in check_cols:
            q1 = full_df[col].quantile(0.25)
            q3 = full_df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            iqr_bounds[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
            }

        outlier_mask = pd.DataFrame(False, index=full_df.index, columns=check_cols)

        for col in check_cols:
            lower = iqr_bounds[col]["lower"]
            upper = iqr_bounds[col]["upper"]
            valid = full_df[[col, target_col]].dropna()
            corr = np.nan

            if len(valid) >= 2:
                corr = _safe_pearson_corr(valid[col], valid[target_col])

            if pd.isna(corr) or abs(corr) < directional_corr_threshold:
                continue
            if corr < 0:
                outlier_mask[col] = full_df[col] < lower
            else:
                outlier_mask[col] = full_df[col] > upper

        outlier_rows = full_df[outlier_mask.any(axis=1)].copy()
        outlier_rows["INDEX"] = outlier_rows.index
        outlier_rows = _normalize_index_result(outlier_rows.drop_duplicates(subset=["INDEX"]))
        outlier_detail_list = []

        for col in check_cols:
            col_mask = outlier_mask[col].fillna(False)
            if not col_mask.any():
                continue

            detail_df = full_df.loc[col_mask, ["DATE", col]].copy()
            detail_df["INDEX"] = detail_df.index
            detail_df = detail_df.rename(columns={col: "VALUE"})
            detail_df["FEATURE"] = col
            outlier_detail_list.append(detail_df[["INDEX", "DATE", "FEATURE", "VALUE"]])

        if outlier_detail_list:
            outlier_detail_df = _normalize_index_result(pd.concat(outlier_detail_list, ignore_index=True))
        else:
            outlier_detail_df = pd.DataFrame(columns=["INDEX", "DATE", "FEATURE", "VALUE"])

        print(
            "IQR 기반 이상치 탐지 결과: "
            f"corr_threshold={directional_corr_threshold}, "
            f"총 {len(outlier_rows)}개 행이 이상치로 감지"
        )

        if return_details:
            return outlier_rows, outlier_detail_df
        return outlier_rows

    @staticmethod
    def anchor_window_date_trend(
        full_df: pd.DataFrame,
        check_cols: list,
        target_col: str,
        context_days: int = 2,
        min_window_points: int = 4,
        min_anchor_rate_quantile: float = 0.8,
        min_level_score: float = 0.5,
        min_sign_agreement: float = 0.5,
    ) -> pd.DataFrame:
        """기존 anchor 기반 날짜별 트렌드 분석 함수."""

        date_col = "DATE"

        def add_series_features(daily_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
            daily_df = daily_df.sort_values(date_col).copy()
            mean_value = daily_df[value_name].mean()
            std_value = daily_df[value_name].std()
            if pd.isna(std_value) or std_value == 0:
                daily_df["level_zscore"] = 0.0
            else:
                daily_df["level_zscore"] = (daily_df[value_name] - mean_value) / std_value

            daily_df["delta"] = daily_df[value_name].diff()
            daily_df["trend_direction"] = np.where(
                daily_df["delta"] > 0,
                "up",
                np.where(daily_df["delta"] < 0, "down", None),
            )
            return daily_df

        def extract_window_pattern(
            daily_df: pd.DataFrame,
            value_name: str,
            center_date: pd.Timestamp,
        ) -> pd.DataFrame:
            window_start = center_date - pd.Timedelta(days=context_days)
            window_end = center_date + pd.Timedelta(days=context_days)
            window_df = daily_df[
                (daily_df[date_col] >= window_start) &
                (daily_df[date_col] <= window_end)
            ][[date_col, value_name]].copy()
            if len(window_df) < min_window_points:
                return pd.DataFrame(columns=[date_col, "delta"])

            window_df["delta"] = window_df[value_name].diff()
            return window_df.dropna(subset=["delta"])[[date_col, "delta"]].reset_index(drop=True)

        defect_daily_df = (
            full_df.groupby(date_col)[target_col]
            .mean()
            .reset_index(name="defect_rate")
        )
        defect_daily_df = add_series_features(defect_daily_df, "defect_rate")
        anchor_threshold = defect_daily_df["defect_rate"].quantile(min_anchor_rate_quantile)
        defect_anchor_df = defect_daily_df[
            (defect_daily_df["defect_rate"] >= anchor_threshold) &
            (defect_daily_df["defect_rate"] > 0)
        ].copy()
        if defect_anchor_df.empty:
            print("날짜별 high defect 분석 결과: defect_rate 높은 anchor 날짜를 찾지 못함")
            return _empty_date_feature_result()

        result_list = []

        feature_daily_map = {
            value_col: add_series_features(
                full_df.groupby(date_col)[value_col].mean().reset_index(name="daily_mean"),
                "daily_mean",
            )
            for value_col in check_cols
        }

        for defect_row in defect_anchor_df.itertuples(index=False):
            center_date = defect_row.DATE
            defect_pattern = extract_window_pattern(defect_daily_df, "defect_rate", center_date)
            if defect_pattern.empty:
                continue

            defect_direction = "up" if defect_pattern["delta"].sum() > 0 else "down"

            for value_col, daily_df in feature_daily_map.items():
                valid = daily_df.merge(
                    defect_daily_df[[date_col, "defect_rate"]],
                    on=date_col,
                    how="inner",
                )
                if len(valid) < min_window_points:
                    continue

                corr = _safe_pearson_corr(valid["daily_mean"], valid["defect_rate"])
                align_sign = -1 if pd.notna(corr) and corr < 0 else 1
                feature_state = "low" if align_sign < 0 else "high"

                feature_pattern = extract_window_pattern(daily_df, "daily_mean", center_date)
                if feature_pattern.empty:
                    continue

                merged_pattern = defect_pattern.merge(
                    feature_pattern,
                    on=date_col,
                    how="inner",
                    suffixes=("_defect", "_feature"),
                )
                if len(merged_pattern) < min_window_points - 1:
                    continue

                defect_delta = merged_pattern["delta_defect"]
                feature_delta = merged_pattern["delta_feature"] * align_sign
                if defect_delta.nunique() < 2 or feature_delta.nunique() < 2:
                    pattern_corr = 0.0
                else:
                    pattern_corr = _safe_pearson_corr(defect_delta, feature_delta)
                    if pd.isna(pattern_corr):
                        pattern_corr = 0.0

                sign_agreement = np.sign(defect_delta).eq(np.sign(feature_delta)).mean()
                feature_direction = "up" if feature_delta.sum() > 0 else "down"
                center_feature = daily_df.loc[daily_df[date_col] == center_date].copy()
                if center_feature.empty:
                    continue
                level_score = float(center_feature["level_zscore"].iloc[0] * align_sign)

                if (
                    level_score >= min_level_score and
                    sign_agreement >= min_sign_agreement and
                    feature_direction == defect_direction
                ):
                    result_list.append(
                        pd.DataFrame(
                            {
                                "DATE": [center_date],
                                "FEATURE": [value_col],
                                "defect_rate": [defect_row.defect_rate],
                                "defect_level_zscore": [defect_row.level_zscore],
                                "defect_trend_direction": [defect_direction],
                                "feature_state": [feature_state],
                                "feature_trend_direction": [feature_direction],
                                "feature_level_score": [level_score],
                                "pattern_corr": [pattern_corr],
                                "sign_agreement": [sign_agreement],
                                "window_points": [len(merged_pattern)],
                            }
                        )
                    )

        if result_list:
            result = pd.concat(result_list, ignore_index=True)
        else:
            result = _empty_date_feature_result()

        result = _normalize_date_feature_result(result)

        print(
            "날짜별 high defect 분석 결과: "
            f"defect_rate가 높은 날짜에 feature 값과 트렌드가 함께 맞는 총 {len(result)}개 날짜-변수 조합 감지"
        )
        return result


class FuncStaticV2:
    """v2 가중치 탐지 로직 모음."""

    @staticmethod
    def pchart_remover(
        full_df: pd.DataFrame,
        target_col: str,
        check_cols: list[str],
        sigma_level: float = 3.0,
        min_outlier_features: int = 1,
        min_subgroup_size: int | None = None,
        return_details: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """전체 데이터 산포 기준으로 row-level 이상치를 탐지한다."""

        df = _ensure_date_series(full_df, "DATE")
        effective_min_features = (
            int(min_subgroup_size)
            if min_subgroup_size is not None
            else int(min_outlier_features)
        )
        effective_min_features = max(effective_min_features, 1)

        valid_check_cols = [
            col for col in check_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not valid_check_cols:
            empty_rows = df.iloc[0:0].copy()
            empty_details = pd.DataFrame(
                columns=[
                    "INDEX",
                    "DATE",
                    "FEATURE",
                    "VALUE",
                    "robust_zscore",
                    "median",
                    "mad",
                ]
            )
            print("전체 산포 기반 index weighting 결과: 사용할 수 있는 수치형 check_cols가 없습니다.")
            if return_details:
                return empty_rows, empty_details
            return empty_rows

        robust_zscore_map: dict[str, pd.Series] = {}
        feature_stats: dict[str, tuple[float, float]] = {}

        for col in valid_check_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            median = float(series.median()) if series.notna().any() else 0.0
            mad = float((series - median).abs().median()) if series.notna().any() else 0.0

            if mad > 0:
                robust_z = 0.6745 * (series - median) / mad
            else:
                std = float(series.std()) if series.notna().any() else 0.0
                if std > 0:
                    robust_z = (series - float(series.mean())) / std
                else:
                    robust_z = pd.Series(0.0, index=df.index, dtype=float)

            robust_z = robust_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            robust_zscore_map[col] = robust_z
            feature_stats[col] = (median, mad)

        robust_zscore_df = pd.DataFrame(robust_zscore_map, index=df.index)
        outlier_mask = robust_zscore_df.abs() > float(sigma_level)
        outlier_feature_count = outlier_mask.sum(axis=1)
        flagged_index = outlier_feature_count >= effective_min_features

        flagged_rows = df.loc[flagged_index].copy()
        flagged_rows["INDEX"] = flagged_rows.index
        flagged_rows = _normalize_index_result(flagged_rows)

        print(
            "전체 산포 기반 index weighting 결과: "
            f"sigma_level={sigma_level}, min_outlier_features={effective_min_features}, "
            f"총 {int(flagged_index.sum())}개 행 감지"
        )

        if not return_details:
            return flagged_rows

        detail_frames: list[pd.DataFrame] = []
        for col in valid_check_cols:
            col_mask = outlier_mask[col].fillna(False)
            if not col_mask.any():
                continue

            median, mad = feature_stats[col]
            detail_df = df.loc[col_mask, ["DATE", col]].copy()
            detail_df["INDEX"] = detail_df.index
            detail_df["FEATURE"] = col
            detail_df["VALUE"] = detail_df[col]
            detail_df["robust_zscore"] = robust_zscore_df.loc[col_mask, col].to_numpy()
            detail_df["median"] = median
            detail_df["mad"] = mad
            detail_frames.append(
                detail_df[["INDEX", "DATE", "FEATURE", "VALUE", "robust_zscore", "median", "mad"]]
            )

        if detail_frames:
            details = _normalize_index_result(pd.concat(detail_frames, ignore_index=True))
        else:
            details = pd.DataFrame(
                columns=["INDEX", "DATE", "FEATURE", "VALUE", "robust_zscore", "median", "mad"]
            )
        return flagged_rows, details

    @staticmethod
    def _segment_cost(
        prefix_sum: np.ndarray,
        prefix_sq_sum: np.ndarray,
        start: int,
        end: int,
    ) -> float:
        count = end - start
        if count <= 0:
            return 0.0
        total = prefix_sum[end] - prefix_sum[start]
        total_sq = prefix_sq_sum[end] - prefix_sq_sum[start]
        mean = total / count
        cost = total_sq - (2.0 * mean * total) + (count * mean * mean)
        return float(max(cost, 0.0))

    @staticmethod
    def _pelt_mean_shift(values: np.ndarray, penalty: float, min_segment_size: int) -> list[int]:
        """Mean shift에 대한 penalty 기반 change point 탐지."""

        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < max(2, min_segment_size * 2):
            return []

        std = float(np.nanstd(values))
        if std > 0:
            values = (values - float(np.nanmean(values))) / std
        else:
            values = values - float(np.nanmean(values))

        prefix_sum = np.zeros(n + 1, dtype=float)
        prefix_sq_sum = np.zeros(n + 1, dtype=float)
        prefix_sum[1:] = np.cumsum(values)
        prefix_sq_sum[1:] = np.cumsum(values ** 2)

        best_cost = np.full(n + 1, np.inf, dtype=float)
        last_change = np.full(n + 1, -1, dtype=int)
        best_cost[0] = -penalty

        for end in range(min_segment_size, n + 1):
            candidates = range(0, end - min_segment_size + 1)
            best_score = np.inf
            best_candidate = -1
            for candidate in candidates:
                if end - candidate < min_segment_size:
                    continue
                score = (
                    best_cost[candidate]
                    + FuncStaticV2._segment_cost(prefix_sum, prefix_sq_sum, candidate, end)
                    + penalty
                )
                if score < best_score:
                    best_score = score
                    best_candidate = candidate

            best_cost[end] = best_score
            last_change[end] = best_candidate

        change_points: list[int] = []
        cursor = n
        while cursor > 0 and last_change[cursor] >= 0:
            start = int(last_change[cursor])
            if start == 0:
                break
            change_points.append(start)
            cursor = start

        return sorted(set(cp for cp in change_points if 0 < cp < n))

    @staticmethod
    def _prepare_daily_feature_means(
        full_df: pd.DataFrame,
        check_cols: list[str],
        date_col: str = "DATE",
    ) -> pd.DataFrame:
        return (
            _ensure_date_series(full_df, date_col)
            .groupby(date_col)[check_cols]
            .mean()
            .reset_index()
            .sort_values(date_col)
            .reset_index(drop=True)
        )

    @staticmethod
    def pelt_cpd_date_trend(
        full_df: pd.DataFrame,
        check_cols: list,
        target_col: str,
        penalty_scale: float = 1.0,
        min_segment_size: int = 3,
        change_point_tolerance_days: int = 1,
        min_effect_size: float = 0.0,
    ) -> pd.DataFrame:
        """PELT 기반으로 defect rate와 feature mean의 동시 change point를 탐지한다."""

        date_col = "DATE"
        df = _ensure_date_series(full_df, date_col)

        defect_daily_df = (
            df.groupby(date_col)[target_col]
            .mean()
            .reset_index(name="defect_rate")
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        if len(defect_daily_df) < max(2, min_segment_size * 2):
            print("PELT 기반 date weighting 결과: 분석 가능한 날짜 수가 부족합니다.")
            return _empty_date_feature_result()

        defect_values = defect_daily_df["defect_rate"].to_numpy(dtype=float)
        defect_std = float(np.nanstd(defect_values))
        defect_penalty = penalty_scale * np.log(len(defect_values)) * max(defect_std, 1e-6)
        defect_cp_idx = FuncStaticV2._pelt_mean_shift(defect_values, defect_penalty, min_segment_size)
        if not defect_cp_idx:
            print("PELT 기반 date weighting 결과: defect rate change point를 찾지 못했습니다.")
            return _empty_date_feature_result()

        feature_daily_df = FuncStaticV2._prepare_daily_feature_means(df, check_cols, date_col=date_col)
        result_rows: list[dict[str, object]] = []

        for feature in check_cols:
            series = feature_daily_df[feature]
            if not pd.api.types.is_numeric_dtype(series) or series.notna().sum() < max(2, min_segment_size * 2):
                continue

            merged = defect_daily_df[[date_col, "defect_rate"]].merge(
                feature_daily_df[[date_col, feature]].rename(columns={feature: "feature_mean"}),
                on=date_col,
                how="inner",
            ).dropna(subset=["feature_mean"])
            if len(merged) < max(2, min_segment_size * 2):
                continue

            feature_values = merged["feature_mean"].to_numpy(dtype=float)
            feature_std = float(np.nanstd(feature_values))
            feature_penalty = penalty_scale * np.log(len(feature_values)) * max(feature_std, 1e-6)
            feature_cp_idx = FuncStaticV2._pelt_mean_shift(feature_values, feature_penalty, min_segment_size)
            if not feature_cp_idx:
                continue

            corr = _safe_pearson_corr(merged["feature_mean"], merged["defect_rate"])

            for defect_cp in defect_cp_idx:
                defect_cp_date = merged.iloc[defect_cp][date_col]

                nearest_feature_cp = min(
                    feature_cp_idx,
                    key=lambda idx: abs((merged.iloc[idx][date_col] - defect_cp_date).days),
                )
                day_gap = abs((merged.iloc[nearest_feature_cp][date_col] - defect_cp_date).days)
                if day_gap > change_point_tolerance_days:
                    continue

                left_start = max(0, defect_cp - min_segment_size)
                right_end = min(len(merged), defect_cp + min_segment_size)
                if defect_cp <= left_start or defect_cp >= right_end:
                    continue

                defect_before = merged.iloc[left_start:defect_cp]["defect_rate"].mean()
                defect_after = merged.iloc[defect_cp:right_end]["defect_rate"].mean()
                feature_before = merged.iloc[left_start:nearest_feature_cp]["feature_mean"].mean()
                feature_after = merged.iloc[nearest_feature_cp:right_end]["feature_mean"].mean()

                defect_shift = defect_after - defect_before
                feature_shift = feature_after - feature_before
                effect_size = abs(feature_after - feature_before)

                if defect_shift <= 0 or effect_size < min_effect_size:
                    continue

                result_rows.append(
                    {
                        "DATE": defect_cp_date,
                        "FEATURE": feature,
                        "defect_shift": float(defect_shift),
                        "feature_shift": float(feature_shift),
                        "feature_effect_size": float(effect_size),
                        "day_gap": int(day_gap),
                        "defect_change_index": int(defect_cp),
                        "feature_change_index": int(nearest_feature_cp),
                        "feature_defect_corr": float(corr) if pd.notna(corr) else np.nan,
                    }
                )

        if not result_rows:
            result = _empty_date_feature_result()
        else:
            result = (
                pd.DataFrame(result_rows)
                .sort_values(
                    ["DATE", "FEATURE", "day_gap", "feature_effect_size"],
                    ascending=[True, True, True, False],
                )
                .drop_duplicates(subset=["DATE", "FEATURE"])
                .reset_index(drop=True)
            )

        result = _normalize_date_feature_result(result)

        print(
            "PELT 기반 date weighting 결과: "
            f"defect change point {len(defect_cp_idx)}개 기준으로 총 {len(result)}개 날짜-변수 조합 감지"
        )
        return result


def iqr_remover(*args, **kwargs):
    return FuncStaticV1.iqr_remover(*args, **kwargs)


def anchor_window_date_trend(*args, **kwargs):
    return FuncStaticV1.anchor_window_date_trend(*args, **kwargs)


def pchart_remover(*args, **kwargs):
    return FuncStaticV2.pchart_remover(*args, **kwargs)


def pelt_cpd_date_trend(*args, **kwargs):
    return FuncStaticV2.pelt_cpd_date_trend(*args, **kwargs)
