import numpy as np
import pandas as pd
import unicodedata
from difflib import SequenceMatcher


def explore_target(y, currency=None):
    """
    Builds an exploratory summary of the target variable.

    Arguments:
        y (pd.Series): target variable
        currency (pd.Series | None): currency values aligned with the target

    Returns:
        pd.DataFrame: target summary overall and, if provided, by currency
    """
    rows = []

    target = pd.to_numeric(y, errors="coerce")

    if currency is None:
        groups = [("All", target)]
    else:
        target_with_currency = pd.DataFrame({
            "target": target,
            "currency": currency
        })

        groups = [("All", target_with_currency["target"])]

        for currency_value, group_df in target_with_currency.groupby("currency", dropna=False):
            groups.append((currency_value, group_df["target"]))

    for group_name, group_target in groups:
        non_missing = group_target.dropna()

        rows.append({
            "group": group_name,
            "count": group_target.count(),
            "missing": group_target.isna().sum(),
            "missing_%": round(group_target.isna().mean() * 100, 2),
            "min": non_missing.min() if len(non_missing) else np.nan,
            "q1": non_missing.quantile(0.25) if len(non_missing) else np.nan,
            "median": non_missing.median() if len(non_missing) else np.nan,
            "mean": non_missing.mean() if len(non_missing) else np.nan,
            "q3": non_missing.quantile(0.75) if len(non_missing) else np.nan,
            "max": non_missing.max() if len(non_missing) else np.nan,
            "std": non_missing.std() if len(non_missing) else np.nan,
            "zero_count": (group_target == 0).sum(),
        })

    return pd.DataFrame(rows)


def format_value_counts(value_counts, max_items):
    """
    Formats category counts into a readable string.

    Arguments:
        value_counts (pd.Series): category counts sorted by frequency
        max_items (int): maximum number of categories to show

    Returns:
        str: formatted category counts
    """
    if value_counts.empty:
        return np.nan

    shown_values = value_counts.head(max_items)

    return " | ".join(
        f"{category}: {count}"
        for category, count in shown_values.items()
    )


def explore_features(df, top_n_categories=5, rare_threshold=1):
    """
    Builds separate exploratory summaries for numeric and categorical features.

    Arguments:
        df (pd.DataFrame): dataset to summarize column by column
        top_n_categories (int): number of categories to show in preview columns
        rare_threshold (int): maximum frequency used to count rare categories

    Returns:
        dict[str, pd.DataFrame]: numeric and categorical feature summaries
    """
    numeric_rows = []
    categorical_rows = []
    row_count = len(df)

    # Ignore CSV-generated index columns such as "Unnamed: 0".
    columns = [
        column for column in df.columns
        if not column.startswith("Unnamed:")
    ]

    for column in columns:
        series = df[column]
        non_missing = series.dropna()
        missing_count = series.isna().sum()
        unique_count = series.nunique(dropna=True)

        base_row = {
            "feature": column,
            "dtype": str(series.dtype),
            "non_missing": series.count(),
            "missing": missing_count,
            "missing_%": round(missing_count / row_count * 100, 2) if row_count else np.nan,
            "unique": unique_count,
            "unique_%": round(unique_count / row_count * 100, 2) if row_count else np.nan,
        }

        if pd.api.types.is_numeric_dtype(series):
            # Numeric features get range, center, dispersion, and outlier checks.
            q1 = non_missing.quantile(0.25) if len(non_missing) else np.nan
            q3 = non_missing.quantile(0.75) if len(non_missing) else np.nan
            iqr = q3 - q1 if len(non_missing) else np.nan

            lower_bound = q1 - 1.5 * iqr if len(non_missing) else np.nan
            upper_bound = q3 + 1.5 * iqr if len(non_missing) else np.nan

            outlier_mask = (
                (series < lower_bound) | (series > upper_bound)
                if len(non_missing)
                else pd.Series(False, index=series.index)
            )

            numeric_rows.append({
                **base_row,
                "min": non_missing.min() if len(non_missing) else np.nan,
                "max": non_missing.max() if len(non_missing) else np.nan,
                "mean": non_missing.mean() if len(non_missing) else np.nan,
                "median": non_missing.median() if len(non_missing) else np.nan,
                "std": non_missing.std() if len(non_missing) else np.nan,
                "q1": q1,
                "q3": q3,
                "zero_count": (series == 0).sum(),
                "outlier_count": outlier_mask.sum(),
                "outlier_%": round(outlier_mask.mean() * 100, 2),
            })

        else:
            value_counts = series.value_counts(dropna=True)
            least_frequent = value_counts.tail(top_n_categories).sort_values()

            # Categorical features get category previews and frequency summaries.
            categorical_rows.append({
                **base_row,
                "categories_preview": format_value_counts(value_counts, top_n_categories),
                "most_frequent": value_counts.index[0] if not value_counts.empty else np.nan,
                "most_frequent_count": value_counts.iloc[0] if not value_counts.empty else np.nan,
                "most_frequent_%": (
                    round(value_counts.iloc[0] / row_count * 100, 2)
                    if row_count and not value_counts.empty
                    else np.nan
                ),
                "least_frequent_preview": format_value_counts(least_frequent, top_n_categories),
                "rare_categories": (value_counts <= rare_threshold).sum(),
            })

    numeric_columns = [
        "feature",
        "dtype",
        "non_missing",
        "missing",
        "missing_%",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "q1",
        "q3",
        "zero_count",
        "outlier_count",
        "outlier_%",
        "unique",
        "unique_%",
    ]

    categorical_columns = [
        "feature",
        "dtype",
        "categories_preview",
        "most_frequent",
        "most_frequent_count",
        "most_frequent_%",
        "least_frequent_preview",
        "rare_categories",
        "missing",
        "missing_%",
        "non_missing",
        "unique",
        "unique_%",
    ]

    numeric_summary = pd.DataFrame(numeric_rows)
    categorical_summary = pd.DataFrame(categorical_rows)

    return {
        "numeric": numeric_summary.reindex(columns=numeric_columns),
        "categorical": categorical_summary.reindex(columns=categorical_columns),
    }


def duplicate_rows_summary(df):
    """ Returns a summary of duplicated rows """

    duplicates = df.duplicated().sum()
    return pd.DataFrame({
        "total_rows": [len(df)],
        "duplicate_rows": [duplicates]
    })


def missing_values_summary(df):
    """Returns the number and percentage of missing values per column."""

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean().values * 100).round(2)
    })

    summary = summary[summary["missing_count"] > 0]
    summary = summary.sort_values("missing_percentage", ascending=False)

    return summary


def unique_values_summary(df):
    """Returns the number and percentage of unique values per column."""

    summary = pd.DataFrame({
        "column": df.columns,
        "unique_values": df.nunique(dropna=True).values,
        "unique_pct": (
            df.nunique(dropna=True).values / len(df) * 100
        ).round(2)
    })

    return summary.sort_values(
        "unique_values",
        ascending=False
    )


def normalize_category_text(value):
    """
    Normalizes categorical text to make similar values easier to compare.

    Returns:
        str: normalized version of the category.
    """
    if pd.isna(value):
        return "missing"

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))

    value = value.replace("_", " ").replace("-", " ")
    value = " ".join(value.split())

    return value


def find_semantic_repetitions(df, columns, similarity_threshold=0.7):
    """
    Finds groups of categories that are semantically similar or almost identical.

    Returns:
        pd.DataFrame: table with feature, grouped original values,
        normalized representative value and total count.
    """
    rows = []

    for column in columns:
        value_counts = df[column].dropna().astype(str).value_counts()

        categories = value_counts.index.tolist()
        normalized_values = {category: normalize_category_text(category) for category in categories}

        used_categories = set()

        for category in categories:
            if category in used_categories:
                continue

            group = [category]
            used_categories.add(category)

            for other_category in categories:
                if other_category in used_categories:
                    continue

                similarity = SequenceMatcher(None, normalized_values[category], normalized_values[other_category]).ratio()

                if similarity >= similarity_threshold:
                    group.append(other_category)
                    used_categories.add(other_category)

            if len(group) > 1:
                normalized_group = sorted({normalized_values[value] for value in group})

                rows.append({
                    "feature": column,
                    "similar_values": " | ".join(group),
                    "normalized_values": " | ".join(normalized_group),
                    "total_count": value_counts[group].sum(),
                    "n_values_grouped": len(group),
                })

    return (pd.DataFrame(rows).sort_values(["feature", "total_count"], ascending=[True, False]).reset_index(drop=True))


def get_constant_columns(df):
    """ Returns constant columns and their unique value """

    constant_columns = []

    for col in df.columns:
        unique_values = df[col].dropna().unique()

        if len(unique_values) <= 1:
            constant_columns.append({
                "column": col,
                "unique_value": unique_values[0] if len(unique_values) == 1 else None
            })

    return pd.DataFrame(constant_columns)


def drop_irrelevant_columns(df, columns_to_drop):
    """ Removes columns that do not provide useful information for the analysis or model """
    df = df.copy()
    return df.drop(columns=columns_to_drop, errors="ignore")


def remove_invalid_values(df, year_col="Año", doors_col="Puertas", max_year=2025, max_doors=5):
    """ Removes rows with invalid year or door values based on predefined limits """
    df = df.copy()
    df = df[df[year_col] <= max_year]
    df = df[df[doors_col] <= max_doors]
    return df


def convert_peso_prices_to_usd(df, price_col="Precio", currency_col="Moneda", peso_symbol="$", exchange_rate=(895.25 + 913) / 2, copy=True,):
    """
    Converts prices in Argentine pesos to USD and removes the currency column.

    Arguments:
        df (pd.DataFrame): dataset with price and currency columns
        price_col (str): price column name
        currency_col (str): currency column name
        peso_symbol (str): value used to identify Argentine pesos
        exchange_rate (float): ARS/USD rate used for conversion
        copy (bool): whether to return a copy instead of modifying df in place

    Returns:
        pd.DataFrame: dataset with peso prices converted to USD and without currency_col
    """
    data = df.copy() if copy else df

    peso_mask = data[currency_col].eq(peso_symbol)
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data.loc[peso_mask, price_col] = data.loc[peso_mask, price_col] / exchange_rate

    return data.drop(columns=[currency_col])


def invert_category_map(category_map):
    """
    Converts a dictionary of final_value -> list of variants
    into variant -> final_value.
    """
    inverted_map = {}

    for final_value, variants in category_map.items():
        final_value_norm = normalize_category_text(final_value)

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            inverted_map[variant_norm] = final_value_norm

    return inverted_map


def apply_semantic_mapping(df, column, category_map):
    """
    Normalizes a categorical column and replaces equivalent values
    using a manual semantic mapping.

    Returns:
        pd.DataFrame: dataset with the cleaned categorical column.
    """
    data = df.copy()

    inverted_map = invert_category_map(category_map)

    normalized_column = data[column].apply(normalize_category_text)

    data[column] = normalized_column.map(inverted_map).fillna(normalized_column)

    return data


def encode_camera_retroceso(df, col="Con cámara de retroceso"):
    data = df.copy()

    data[col] = (
        data[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"sí": 1, "si": 1, "no": 0})
    )

    return data




def one_hot_encoding(df, categorical_cols = ["escuela", "semestre"], train = True, categories_map = None):
    """
    Applies one-hot encoding to categorical columns.

    Arguments:
        df (pd.DataFrame): dataset to encode
        categorical_cols (list[str]): categorical columns to encode
        train (bool): whether the dataset is a training set
        categories_map (dict): categories learned from the training set

    Returns:
        pd.DataFrame: one-hot encoded dataset
        dict: [only if train=True] categories used for encoding
    """
    # During training, learn the categories and create new columns for each category
    if train:
        categories_map = {}

        for col in categorical_cols:
            categories_map[col] = sorted(df[col].dropna().unique())

            for category in categories_map[col]:
                df[f"{col}{category}"] = (df[col] == category).astype(int)

            df = df.drop(col, axis=1)

        return df, categories_map

    # During validation/test, use the learned categories to create the same columns, filling with 0 for unseen categories
    for col in categorical_cols:
        for category in categories_map[col]:
            df[f"{col}{category}"] = (df[col] == category).astype(int)

        df = df.drop(col, axis=1)

    return df
