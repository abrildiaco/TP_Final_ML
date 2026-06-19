import unicodedata # Library for normalizing text, used in semantic repetition detection
from difflib import SequenceMatcher # Library for measuring string similarity, used in semantic repetition detection
import re # Library for regular expressions, used in text normalization and feature extraction

import numpy as np
import pandas as pd

# ========================= Label Analysis =========================
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
            "currency": currency,
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


# ========================= Feature Analysis =========================
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

    # Ignore CSV-generated index columns
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
            # Numeric features get range, center, dispersion, and outlier checks
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

            # Categorical features get category previews and frequency summaries
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
        "feature", "dtype", "non_missing", "missing", "missing_%",
        "min", "max", "mean", "median", "std", "q1", "q3",
        "zero_count", "outlier_count", "outlier_%", "unique", "unique_%",
    ]

    categorical_columns = [
        "feature", "dtype", "categories_preview", "most_frequent",
        "most_frequent_count", "most_frequent_%", "least_frequent_preview",
        "rare_categories", "missing", "missing_%", "non_missing",
        "unique", "unique_%",
    ]

    return {
        "numeric": pd.DataFrame(numeric_rows).reindex(columns=numeric_columns),
        "categorical": pd.DataFrame(categorical_rows).reindex(columns=categorical_columns),
    }


def duplicate_rows_summary(df):
    """
    Returns a summary of duplicated rows.

    Arguments:
        df (pd.DataFrame): dataset to analyze

    Returns:
        pd.DataFrame: duplicated row summary
    """
    return pd.DataFrame({
        "total_rows": [len(df)],
        "duplicate_rows": [df.duplicated().sum()],
    })


def missing_values_summary(df):
    """
    Returns the number and percentage of missing values per column.

    Arguments:
        df (pd.DataFrame): dataset to analyze

    Returns:
        pd.DataFrame: missing values summary
    """
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean().values * 100).round(2),
    })

    summary = summary[summary["missing_count"] > 0]

    return summary.sort_values("missing_percentage", ascending=False)


def unique_values_summary(df):
    """
    Returns the number and percentage of unique values per column.

    Arguments:
        df (pd.DataFrame): dataset to analyze

    Returns:
        pd.DataFrame: unique values summary
    """
    summary = pd.DataFrame({
        "column": df.columns,
        "unique_values": df.nunique(dropna=True).values,
        "unique_pct": (df.nunique(dropna=True).values / len(df) * 100).round(2),
    })

    return summary.sort_values("unique_values", ascending=False)


def normalize_category_text(value):
    """
    Normalizes categorical text to make similar values easier to compare.

    Arguments:
        value (object): original category value

    Returns:
        str: normalized category value
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
    Finds groups of categories that are similar or almost identical.

    Arguments:
        df (pd.DataFrame): dataset containing categorical columns
        columns (list[str]): categorical columns to inspect
        similarity_threshold (float): minimum similarity score used to group values

    Returns:
        pd.DataFrame: similar category groups by feature
    """
    rows = []

    for column in columns:
        value_counts = df[column].dropna().astype(str).value_counts()
        categories = value_counts.index.tolist()

        normalized_values = {
            category: normalize_category_text(category)
            for category in categories
        }

        used_categories = set()

        for category in categories:
            if category in used_categories:
                continue

            group = [category]
            used_categories.add(category)

            for other_category in categories:
                if other_category in used_categories:
                    continue
                
                # Calculate similarity between normalized category values
                similarity = SequenceMatcher(
                    None,
                    normalized_values[category],
                    normalized_values[other_category],
                ).ratio()

                if similarity >= similarity_threshold:
                    group.append(other_category)
                    used_categories.add(other_category)

            if len(group) > 1:
                normalized_group = sorted({
                    normalized_values[value]
                    for value in group
                })

                rows.append({
                    "feature": column,
                    "similar_values": " | ".join(group),
                    "normalized_values": " | ".join(normalized_group),
                    "total_count": value_counts[group].sum(),
                    "n_values_grouped": len(group),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "feature", "similar_values", "normalized_values",
            "total_count", "n_values_grouped",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "total_count"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ========================= Utils for Initial Preprocessing =========================
def get_constant_columns(df):
    """
    Returns columns with one or zero non-missing unique values.

    Arguments:
        df (pd.DataFrame): dataset to analyze

    Returns:
        pd.DataFrame: constant columns and their unique values
    """
    constant_columns = []

    for column in df.columns:
        unique_values = df[column].dropna().unique()

        if len(unique_values) <= 1:
            constant_columns.append({
                "column": column,
                "unique_value": unique_values[0] if len(unique_values) == 1 else None,
            })

    return pd.DataFrame(constant_columns)


# REVISAR SI LA QUEREMOS BORRAR
def models_by_brand(df, brand, brand_col = "Marca", model_col = "Modelo"):
    """
    Returns the available models for a selected brand.

    Arguments:
        df (pd.DataFrame): dataset containing brand and model columns
        brand (str): brand to inspect
        brand_col (str): brand column name
        model_col (str): model column name

    Returns:
        pd.DataFrame: models for the selected brand with counts and percentages
    """
    brand_data = df[
        df[brand_col].astype(str).str.lower().str.strip()
        == str(brand).lower().strip()
    ]

    model_counts = (
        brand_data[model_col]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    model_counts.columns = [model_col, "count"]

    if len(brand_data) > 0:
        model_counts["percentage"] = (
            model_counts["count"] / len(brand_data) * 100
        ).round(2)
    else:
        model_counts["percentage"] = []

    return model_counts


def invert_category_map(category_map):
    """
    Converts a final-value mapping into a normalized variant mapping.

    Arguments:
        category_map (dict): dictionary with final values as keys and variants as values

    Returns:
        dict: normalized variant to normalized final value mapping
    """
    inverted_map = {}

    for final_value, variants in category_map.items():
        final_value_norm = normalize_category_text(final_value)

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            inverted_map[variant_norm] = final_value_norm

    return inverted_map


def compact_value_counts(df, columns):
    """
    Builds a compact table with the frequency of every category
    for multiple categorical features.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        columns (list[str]): categorical columns

    Returns:
        pd.DataFrame: compact frequency table.
    """
    tables = []

    for column in columns:
        counts = (df[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count"))

        counts.columns = [
            f"{column}",
            f"{column}_count",
        ]

        counts = counts.reset_index(drop=True)

        tables.append(counts)

    return pd.concat(tables, axis=1)


import re


def extract_engine_liters(value):
    """
    Extracts engine displacement in liters from a text value.

    Arguments:
        value (object): original engine value

    Returns:
        float: engine displacement in liters or np.nan if not found
    """
    if pd.isna(value):
        return np.nan

    text = normalize_category_text(value)
    text = text.replace(",", ".")

    # Separate numbers from letters
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    for number_text in numbers:
        number = float(number_text)

        # Values like 1.4, 2.0, 3.6 usually represent liters
        if 0.8 <= number <= 8.0:
            return round(number, 1)

        # Values like 1400, 1600, 2000 usually represent cubic centimeters
        if 800 <= number <= 8000:
            return round(number / 1000, 1)

    return np.nan


def encode_engine_size(engine_liters):
    """
    Encodes engine displacement into an ordinal numeric group.

    Arguments:
        engine_liters (float): engine displacement in liters

    Returns:
        int: encoded engine size group
    """
    if pd.isna(engine_liters):
        return 0

    if engine_liters <= 1.2:
        return 1

    if engine_liters <= 1.6:
        return 2

    if engine_liters <= 2.0:
        return 3

    if engine_liters <= 2.8:
        return 4

    return 5


def has_turbo(value, turbo_patterns):
    """
    Detects whether an engine text suggests turbo.

    Arguments:
        value (object): original engine value
        turbo_patterns (list[str]): regex patterns that indicate turbo

    Returns:
        int: 1 if turbo is detected, 0 otherwise
    """
    if pd.isna(value):
        return 0

    text = normalize_category_text(value)
    pattern = "|".join(turbo_patterns)

    return int(bool(re.search(pattern, text)))


def add_engine_numeric_features(df, engine_col="Motor", turbo_patterns=None):
    """
    Creates numeric engine features from a raw engine text column.

    Arguments:
        df (pd.DataFrame): dataset containing the engine column
        engine_col (str): raw engine column name
        turbo_patterns (list[str] | None): regex patterns that indicate turbo

    Returns:
        pd.DataFrame: dataset with numeric engine features
    """
    data = df.copy()

    data["engine_liters"] = data[engine_col].apply(extract_engine_liters)
    data["engine_size_group_code"] = data["engine_liters"].apply(encode_engine_size)
    data["engine_has_turbo"] = data[engine_col].apply(
        lambda value: has_turbo(value, turbo_patterns=turbo_patterns)
    )

    return data


def drop_irrelevant_columns(df, columns_to_drop):
    """
    Removes columns that do not provide useful information.

    Arguments:
        df (pd.DataFrame): dataset to transform
        columns_to_drop (list[str]): columns to remove

    Returns:
        pd.DataFrame: dataset without selected columns
    """
    data = df.copy()

    return data.drop(columns=columns_to_drop, errors="ignore")


def remove_invalid_values(df, range_rules, copy=True):
    """
    Filters rows using numeric range rules.

    Arguments:
        df (pd.DataFrame): dataset to filter
        range_rules (dict): column names mapped to minimum and maximum valid values
        copy (bool): whether to return a copy instead of modifying df in place

    Returns:
        pd.DataFrame: dataset with rows inside the selected ranges
    """
    data = df.copy() if copy else df

    for column, limits in range_rules.items():
        min_value = limits.get("min", -np.inf)
        max_value = limits.get("max", np.inf)

        values = pd.to_numeric(data[column], errors="coerce")
        data = data[(values >= min_value) & (values <= max_value)]

    return data


def convert_peso_prices_to_usd(df, price_col = "Precio", currency_col = "Moneda", peso_symbol = "$",
                               exchange_rate = (895.25 + 913) / 2, copy = True,):
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
        pd.DataFrame: dataset with peso prices converted to USD
    """
    data = df.copy() if copy else df

    peso_mask = data[currency_col].eq(peso_symbol)
    data[price_col] = pd.to_numeric(data[price_col])
    data.loc[peso_mask, price_col] = data.loc[peso_mask, price_col] / exchange_rate

    return data.drop(columns = [currency_col])


def extract_first_integer(value):
    """
    Extracts the first space-separated value and converts it to integer.

    Arguments:
        value (object): value to transform

    Returns:
        int | float: extracted integer or np.nan if conversion is not possible
    """
    if pd.isna(value):
        return np.nan

    first_part = str(value).strip().split()[0]
    first_part = first_part.replace(",", "")

    try:
        return int(float(first_part))
    except ValueError:
        return np.nan


def map_column_values(df, column, value_map, copy = True):
    """
    Maps values from a column using a dictionary.

    Arguments:
        df (pd.DataFrame): dataset to transform
        column (str): column to map
        value_map (dict): original value to mapped value dictionary
        copy (bool): whether to return a copy instead of modifying df in place

    Returns:
        pd.DataFrame: dataset with mapped column values
    """
    data = df.copy() if copy else df

    data[column] = (
        data[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(value_map)
    )

    return data


def one_hot_encoding(df, categorical_cols = None, train=True, categories_map=None):
    """
    Applies one-hot encoding to categorical columns.

    Arguments:
        df (pd.DataFrame): dataset to encode
        categorical_cols (list[str] | None): categorical columns to encode
        train (bool): whether the dataset is a training set
        categories_map (dict | None): categories learned from the training set

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict]: encoded dataset and, during training, learned categories
    """
    data = df.copy()
    categorical_cols = categorical_cols or []

    if train:
        categories_map = {}

        for column in categorical_cols:
            categories_map[column] = sorted(data[column].dropna().unique())

            # Learn categories only from the training data
            for category in categories_map[column]:
                data[f"{column}_{category}"] = (data[column] == category).astype(int)

            data = data.drop(columns=[column])

        return data, categories_map

    for column in categorical_cols:
        # Reuse training categories so validation and test keep the same columns
        for category in categories_map[column]:
            data[f"{column}_{category}"] = (data[column] == category).astype(int)

        data = data.drop(columns=[column])

    return data
>>>>>>> e832c99349da0f641271e50966dce8b0b6f2e340
