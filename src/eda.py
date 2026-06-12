import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def explore_features(df, compact=True):
    """
    Builds a feature-level exploratory summary for a DataFrame.

    Arguments:
        df (pd.DataFrame): dataset to summarize column by column
        compact (bool): whether to return a smaller notebook-friendly summary

    Returns:
        pd.DataFrame: summary with missing values, unique values, descriptive
        statistics, frequent values, and example observations
    """
    rows = []
    row_count = len(df)

    # Ignore CSV-generated index columns such as "Unnamed: 0".
    columns = [
        column for column in df.columns
        if not column.startswith("Unnamed:")
    ]

    for column in columns:
        series = df[column]
        non_missing = series.dropna()
        value_counts = series.value_counts(dropna=True)

        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = series.nunique(dropna=True)

        # Base information useful for every type of feature.
        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "missing": series.isna().sum(),
            "missing_%": round(series.isna().mean() * 100, 2),
            "unique": unique_count,
            "unique_%": round(unique_count / row_count * 100, 2) if row_count else np.nan,
            "most_frequent": value_counts.index[0] if not value_counts.empty else np.nan,
            "most_frequent_count": value_counts.iloc[0] if not value_counts.empty else np.nan,
        }

        if is_numeric:
            # Numeric features get distribution statistics.
            row.update({
                "min": non_missing.min() if len(non_missing) else np.nan,
                "q1": non_missing.quantile(0.25) if len(non_missing) else np.nan,
                "median": non_missing.median() if len(non_missing) else np.nan,
                "mean": non_missing.mean() if len(non_missing) else np.nan,
                "q3": non_missing.quantile(0.75) if len(non_missing) else np.nan,
                "max": non_missing.max() if len(non_missing) else np.nan,
                "std": non_missing.std() if len(non_missing) else np.nan,
                "zero_count": (series == 0).sum(),
            })
        else:
            # Categorical/text features keep examples instead of numeric stats.
            row.update({
                "min": np.nan,
                "median": np.nan,
                "mean": np.nan,
                "max": np.nan,
                "std": np.nan,
                "zero_count": np.nan,
                "examples": ", ".join(non_missing.astype(str).head(3)),
            })

        rows.append(row)

    summary = pd.DataFrame(rows)

    compact_columns = [
        "feature",
        "dtype",
        "missing",
        "missing_%",
        "unique",
        "most_frequent",
        "min",
        "median",
        "mean",
        "max",
    ]

    full_columns = [
        "feature",
        "dtype",
        "missing",
        "missing_%",
        "unique",
        "unique_%",
        "min",
        "q1",
        "median",
        "mean",
        "q3",
        "max",
        "std",
        "zero_count",
        "most_frequent",
        "most_frequent_count",
        "examples",
    ]

    selected_columns = compact_columns if compact else full_columns
    selected_columns = [column for column in selected_columns if column in summary.columns]

    return summary[selected_columns]