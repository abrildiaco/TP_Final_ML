import unicodedata  # Library for normalizing text
from difflib import SequenceMatcher  # Library for measuring string similarity
import re  # For regex operations in engine feature extraction

import numpy as np
import pandas as pd


# =========================  Private Helpers  =========================

def display_table(df):
    """
    Displays a DataFrame in notebooks without showing the default pandas index.

    Arguments:
        df (pd.DataFrame): table to display

    Returns:
        None
    """
    try:
        from IPython.display import display

        display(df.style.hide(axis="index"))
    except ImportError:
        print(df.to_string(index=False))


def _build_missing_mask(series, extra_missing=("missing",)):
    """
    Returns a boolean mask that is True wherever a Series value should be
    considered missing — either a real NaN or a placeholder string like
    "missing". Centralizing this logic means the definition of "missing"
    only needs to change in one place.

    Arguments:
        series (pd.Series): column to evaluate
        extra_missing (tuple[str]): additional string values treated as missing,
            compared in a case-insensitive, stripped manner

    Returns:
        pd.Series[bool]: True where the value is considered missing
    """
    normalized = {v.strip().lower() for v in extra_missing}
    return series.isna() | series.astype(str).str.strip().str.lower().isin(normalized)


# ========================= Target Analysis =========================

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


# ========================= Feature Summaries =========================

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


def missing_values_summary(df, missing_values=("missing",)):
    """
    Returns the number and percentage of missing values per column,
    treating both real NaN and placeholder strings as missing.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        missing_values (tuple[str]): text values treated as missing

    Returns:
        pd.DataFrame: missing values summary, sorted by percentage descending
    """
    rows = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Normalize text columns so placeholder strings are caught by _build_missing_mask
            series = series.apply(
                lambda value: np.nan if pd.isna(value) else normalize_category_text(value)
            )

        missing_count = _build_missing_mask(series, extra_missing=missing_values).sum()

        if missing_count > 0:
            rows.append({
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": round(missing_count / len(df) * 100, 2) if len(df) else np.nan,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("missing_percentage", ascending=False)
        .reset_index(drop=True)
    )


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


# ========================= Semantic Category Inspection =========================

def normalize_category_text(value):
    """
    Normalizes categorical text to make similar values easier to compare.
    Converts to lowercase, removes accents, and collapses irregular spacing
    and separators so that variants like "Blanco", "blanca", and "blanco "
    all map to the same string.

    Arguments:
        value (object): original category value

    Returns:
        str: normalized category value, or "missing" if the input is NaN
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
    Finds groups of categories that are similar or almost identical,
    helping identify variants that should be merged before modeling.

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


def invert_category_map(category_map):
    """
    Converts a canonical-value mapping into a normalized variant-to-canonical
    mapping, ready to be used with Series.map().

    Arguments:
        category_map (dict): canonical values as keys, lists of accepted variants as values

    Returns:
        dict: normalized variant -> normalized canonical value
    """
    inverted_map = {}

    for final_value, variants in category_map.items():
        final_value_norm = normalize_category_text(final_value)

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            inverted_map[variant_norm] = final_value_norm

    return inverted_map


# ========================= Specific Inspection Helpers =========================

def count_category_mentions_in_text(df, target_col, text_cols, category_map, ignored_categories=("missing", "otros"),
                                    only_missing_target=False, only_rows_with_mentions=False):
    """
    Searches for category mentions inside text columns and returns a row-level
    table indicating which categories were found. Used to recover missing values
    from free-text fields like titles and descriptions.

    Arguments:
        df (pd.DataFrame): dataset containing target and text columns
        target_col (str): column used to identify missing rows
        text_cols (tuple[str] | list[str]): text columns to search in
        category_map (dict): canonical values as keys, lists of accepted variants as values
        ignored_categories (tuple[str]): mapped categories excluded from the count
        only_missing_target (bool): if True, searches only rows where target_col is missing
        only_rows_with_mentions (bool): if True, returns only rows with at least one mention

    Returns:
        pd.DataFrame: row-level table with matched categories and mention counts
    """
    variant_to_category = {}
    ignored_normalized = {normalize_category_text(c) for c in ignored_categories}

    for final_value, variants in category_map.items():
        final_norm = normalize_category_text(final_value)

        if final_norm in ignored_normalized:
            continue

        variant_to_category[final_norm] = final_norm

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            variant_to_category[variant_norm] = final_norm

    # Sort longer variants first so "marcha atras" is matched before "marcha"
    variants = sorted(
        variant_to_category,
        key=lambda value: len(value.split()),
        reverse=True
    )

    missing_col = f"{target_col}_is_missing"
    missing_target_mask = _build_missing_mask(df[target_col])

    rows_to_search = df[missing_target_mask] if only_missing_target else df

    rows = []

    for row_index, row in rows_to_search.iterrows():
        text = " ".join(
            str(row[col])
            for col in text_cols
            if col in df.columns and not pd.isna(row[col])
        )

        normalized_text = normalize_category_text(text)

        mentions = []

        for variant in variants:
            pattern = r"\b" + re.escape(variant) + r"\b"
            matches = re.findall(pattern, normalized_text)

            for _ in matches:
                mentions.append(variant_to_category[variant])

        if only_rows_with_mentions and len(mentions) == 0:
            continue

        rows.append({
            "row_index": row_index,
            missing_col: missing_target_mask.loc[row_index],
            "matched_categories": " | ".join(sorted(set(mentions))),
            "n_category_mentions": len(mentions),
        })

    return pd.DataFrame(rows)


def frequent_words_table(df, text_cols, top_n=30, min_word_length=3, stop_words=None):
    """
    Counts the most frequent words across selected text columns.

    Arguments:
        df (pd.DataFrame): dataset containing the text columns
        text_cols (tuple[str] | list[str]): text columns to analyze
        top_n (int): number of words to return
        min_word_length (int): minimum word length to include
        stop_words (set[str] | list[str] | None): words to exclude; if None,
            a small default Spanish stopword list is used

    Returns:
        pd.DataFrame: table with word and count columns
    """
    default_stop_words = {
        "con", "del", "las", "los", "una", "uno", "unos", "unas",
        "para", "por", "que", "sin", "como", "mas", "muy", "esta",
        "este", "estos", "estas", "sobre", "todo", "todas", "todos",
        "desde", "hasta", "solo", "tambien", "sus", "son", "ser",
        "fue", "hay", "tiene", "tienen", "km", "kms", "y", "el", "la",
        "financiacion", "precio", "vehiculo", "vehiculos", "auto",
        "mejor", "cuotas", "entrega", "oficial", "pago",
        "concesionario", "tomamos", "inmediata", "mercado", "valor",
        "anos", "fijas", "stock", "dia"
    }

    if stop_words is None:
        stop_words = default_stop_words
    else:
        stop_words = {normalize_category_text(word) for word in stop_words}

    text = " ".join(
        df[column].dropna().astype(str).str.cat(sep=" ")
        for column in text_cols
        if column in df.columns
    )

    normalized_text = normalize_category_text(text)
    words = re.findall(r"\b[a-z0-9]+\b", normalized_text)

    words = [
        word for word in words
        if len(word) >= min_word_length and word not in stop_words
    ]

    word_counts = pd.Series(words).value_counts().head(top_n)

    return (
        word_counts
        .reset_index()
        .rename(columns={"index": "word", "count": "count"})
    )


def count_interest_terms_in_text(df, text_cols, terms_map):
    """
    Counts selected words or phrases inside text columns.

    Arguments:
        df (pd.DataFrame): dataset containing the text columns
        text_cols (tuple[str] | list[str]): text columns to search in
        terms_map (dict): output labels as keys and words/phrases to search as values

    Returns:
        pd.DataFrame: table with term, count, and row_count columns
    """
    text_source = pd.Series("", index=df.index)

    for column in text_cols:
        if column in df.columns:
            text_source = text_source + " " + df[column].fillna("").astype(str)

    normalized_text = text_source.apply(normalize_category_text)

    rows = []

    for term, variants in terms_map.items():
        if isinstance(variants, str):
            variants = [variants]

        total_count = 0
        matched_rows = pd.Series(False, index=df.index)

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            pattern = r"\b" + re.escape(variant_norm) + r"\b"

            counts = normalized_text.str.count(pattern)
            total_count += counts.sum()
            matched_rows = matched_rows | counts.gt(0)

        rows.append({
            "term": term,
            "count": int(total_count),
            "row_count": int(matched_rows.sum()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["row_count", "count"], ascending=False)
        .reset_index(drop=True)
    )


def models_by_brand(df, brand, brand_col="Marca", model_col="Modelo"):
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


def print_missing_feature_text(df, feature_col, text_cols=("Título", "Descripción"), max_rows=None):
    """
    Prints text columns for rows where a selected feature is missing.
    Useful for manual inspection when deciding how to impute a column.

    Arguments:
        df (pd.DataFrame): dataset containing the selected feature and text columns
        feature_col (str): feature used to detect missing rows
        text_cols (tuple[str]): text columns to print for manual inspection
        max_rows (int | None): maximum number of missing rows to print

    Returns:
        None
    """
    missing_rows = df[_build_missing_mask(df[feature_col])].copy()

    if max_rows is not None:
        missing_rows = missing_rows.head(max_rows)

    print(f"Missing values in '{feature_col}': {len(missing_rows)} rows")
    print("=" * 100)

    for idx, row in missing_rows.iterrows():
        print(f"\nROW INDEX: {idx}\n")

        for col in text_cols:
            if col in df.columns:
                value = row[col]
                if pd.isna(value):
                    value = "Missing"

                print(f"{col}:")
                print(str(value))
                print()

        print("=" * 100)
