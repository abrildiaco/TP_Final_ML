import re

import numpy as np
import pandas as pd

from eda_utils import _normalize_category_text
import preprocessing as prep


DEFAULT_REFERENCE_YEAR = 2025
DEFAULT_ZERO_KM_THRESHOLD = 100
DEFAULT_PREMIUM_BRANDS = [
    "alfa romeo",
    "audi",
    "bmw",
    "land rover",
    "mercedes benz",
    "porsche",
    "volvo",
]
DEFAULT_COLUMNS_TO_DROP = [
    "Título",
    "Descripción",
    "Versión",
]
DEFAULT_CATEGORICAL_COLS = [
    "Marca",
    "Modelo",
    "Marca_Modelo",
    "Color",
    "Tipo de vendedor",
    "Tipo de combustible",
    "Transmisión",
]
DEFAULT_BINARY_MISSING_COLS = [
    "Con cámara de retroceso",
]
ALL_FEATURE_BLOCKS = [
    "usage",
    "brand_model",
    "premium",
    "cilindrada_missing",
]
DEFAULT_FEATURE_VARIANTS = [
    {
        "name": "baseline",
        "feature_blocks": [],
    },
    {
        "name": "usage",
        "feature_blocks": ["usage"],
    },
    {
        "name": "brand_model",
        "feature_blocks": ["brand_model"],
    },
    {
        "name": "premium_brand",
        "feature_blocks": ["premium"],
    },
    {
        "name": "cilindrada_missing",
        "feature_blocks": ["cilindrada_missing"],
    },
    {
        "name": "all_features",
        "feature_blocks": "all",
    },
    {
        "name": "all_features_without_brand_model_originals",
        "feature_blocks": "all",
        "drop_cols": ["Marca", "Modelo"],
    },
]


# ======================== Versión ======================================
def matches_any_pattern(value, patterns):
    """
    Checks whether a text value matches at least one regex pattern.

    The input text is normalized before matching, so patterns should be written
    assuming lowercase text without accents.

    Arguments:
        value (object): raw text value to inspect
        patterns (list[str]): regex patterns to search in the normalized text

    Returns:
        int: 1 if any pattern matches, 0 otherwise
    """
    if pd.isna(value):
        return 0

    text = _normalize_category_text(value)

    return int(any(re.search(pattern, text) for pattern in patterns))


def classify_text_tier(value, tier_patterns, default_tier=1):
    """
    Classifies a text value into an ordinal tier using regex pattern groups.

    Patterns are received as a dictionary where keys are ordinal tier values and
    values are lists of regex patterns. Higher tiers are checked first so more
    specific or premium signals can take priority over general ones.

    Example:
        tier_patterns = {
            3: [r"\\bpremium\\b", r"\\blimited\\b"],
            2: [r"\\bhighline\\b", r"\\bltz\\b"],
            1: [r"\\bcomfortline\\b"],
            0: [r"\\bbase\\b"],
        }

    Arguments:
        value (object): raw text value to classify
        tier_patterns (dict[int, list[str]]): tier values mapped to regex patterns
        default_tier (int): tier assigned when no pattern matches

    Returns:
        int: assigned ordinal tier
    """
    if pd.isna(value):
        return default_tier

    for tier in sorted(tier_patterns.keys(), reverse=True):
        if matches_any_pattern(value, tier_patterns[tier]):
            return tier

    return default_tier


def is_unknown_text_tier(value, tier_patterns):
    """
    Identifies whether a text value did not match any tier pattern.

    Arguments:
        value (object): raw text value to inspect
        tier_patterns (dict[int, list[str]]): tier values mapped to regex patterns

    Returns:
        int: 1 if no tier pattern matches, 0 otherwise
    """
    all_patterns = []

    for patterns in tier_patterns.values():
        all_patterns.extend(patterns)

    return int(not matches_any_pattern(value, all_patterns))


def add_text_pattern_features( df, source_col, tier_patterns=None, binary_patterns=None, numeric_extractors=None, prefix=None, default_tier=1, drop_source_col=False,):
    """
    Adds ordinal, binary and numeric features extracted from a text column.

    Arguments:
        df (pd.DataFrame): dataset containing the source text column
        source_col (str): column used to extract features
        tier_patterns (dict[int, list[str]] | None): ordinal tier patterns
        binary_patterns (dict[str, list[str]] | None): new binary feature names
            mapped to regex patterns
        numeric_extractors (dict[str, callable] | None): new numeric feature names
            mapped to functions that receive a text value and return a number
        prefix (str | None): prefix used for generated feature names. If None,
            source_col is used
        default_tier (int): tier assigned when no tier pattern matches
        drop_source_col (bool): whether to drop the original text column

    Returns:
        pd.DataFrame: dataset with the new extracted features
    """
    data = df.copy()

    if source_col not in data.columns:
        return data

    feature_prefix = prefix or source_col

    if tier_patterns is not None:
        data[f"{feature_prefix}_Tier"] = data[source_col].apply(
            lambda value: classify_text_tier(
                value,
                tier_patterns=tier_patterns,
                default_tier=default_tier,
            )
        )

        data[f"{feature_prefix}_Tier_Unknown"] = data[source_col].apply(
            lambda value: is_unknown_text_tier(value, tier_patterns)
        )

    for feature_name, patterns in (binary_patterns or {}).items():
        data[f"{feature_prefix}_{feature_name}"] = data[source_col].apply(
            lambda value: matches_any_pattern(value, patterns)
        )

    for feature_name, extractor in (numeric_extractors or {}).items():
        data[f"{feature_prefix}_{feature_name}"] = data[source_col].apply(extractor)

    if drop_source_col:
        data = data.drop(columns=[source_col])

    return data


def add_text_pattern_features_to_split(train_df, val_df, source_col,
                                       tier_patterns=None, binary_patterns=None,
                                       numeric_extractors=None, prefix=None,
                                       default_tier=1, drop_source_col=False,
                                       columns_to_drop=None):
    """
    Adds text-pattern features to train and validation datasets consistently.

    This helper keeps notebook experiments cleaner when the same text feature
    extraction must be applied to both splits.

    Arguments:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        source_col (str): column used to extract text features
        tier_patterns (dict[int, list[str]] | None): ordinal tier patterns
        binary_patterns (dict[str, list[str]] | None): binary feature patterns
        numeric_extractors (dict[str, callable] | None): numeric extractors
        prefix (str | None): prefix used for generated feature names
        default_tier (int): tier assigned when no pattern matches
        drop_source_col (bool): whether to drop the original text column
        columns_to_drop (list[str] | None): extra columns removed afterwards

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: transformed train and validation data
    """
    train_data = add_text_pattern_features(
        train_df,
        source_col=source_col,
        tier_patterns=tier_patterns,
        binary_patterns=binary_patterns,
        numeric_extractors=numeric_extractors,
        prefix=prefix,
        default_tier=default_tier,
        drop_source_col=drop_source_col,
    )

    val_data = add_text_pattern_features(
        val_df,
        source_col=source_col,
        tier_patterns=tier_patterns,
        binary_patterns=binary_patterns,
        numeric_extractors=numeric_extractors,
        prefix=prefix,
        default_tier=default_tier,
        drop_source_col=drop_source_col,
    )

    if columns_to_drop:
        train_data = prep.drop_irrelevant_columns(train_data, columns_to_drop)
        val_data = prep.drop_irrelevant_columns(val_data, columns_to_drop)

    return train_data, val_data


# ======================== Initial Feature Engineering ========================

def add_usage_features(df, year_col="Año", km_col="Kilómetros",
                       reference_year=DEFAULT_REFERENCE_YEAR,
                       zero_km_threshold=DEFAULT_ZERO_KM_THRESHOLD):
    """
    Adds vehicle age, kilometers-per-year and zero-kilometer indicators.

    Arguments:
        df (pd.DataFrame): dataset containing year and kilometer columns
        year_col (str): vehicle year column
        km_col (str): kilometers column
        reference_year (int): year used to compute vehicle age
        zero_km_threshold (int | float): maximum kilometers considered as 0km

    Returns:
        pd.DataFrame: dataset with usage-related features
    """
    data = df.copy()

    data[year_col] = pd.to_numeric(data[year_col], errors="coerce")
    data[km_col] = pd.to_numeric(data[km_col], errors="coerce")

    data["Antigüedad"] = (reference_year - data[year_col]).clip(lower=0)
    age_for_ratio = data["Antigüedad"].replace(0, 1)

    data["Kilómetros_por_año"] = data[km_col] / age_for_ratio
    data["Es_0km"] = (data[km_col] <= zero_km_threshold).astype(int)

    return data


def add_brand_model_feature(train_df, val_df, brand_col="Marca", model_col="Modelo",
                            min_count=20):
    """
    Adds a grouped brand-model interaction feature using only train frequencies.

    Rare combinations are grouped as 'other' to avoid creating many sparse
    columns and to keep validation aligned with training categories.

    Arguments:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        brand_col (str): brand column
        model_col (str): model column
        min_count (int): minimum train frequency required to keep a combination

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and validation datasets with
        the brand-model feature added
    """
    train_data = train_df.copy()
    val_data = val_df.copy()

    train_combo = train_data[brand_col].astype(str) + "_" + train_data[model_col].astype(str)
    val_combo = val_data[brand_col].astype(str) + "_" + val_data[model_col].astype(str)

    frequent_combos = train_combo.value_counts()
    frequent_combos = frequent_combos[frequent_combos >= min_count].index

    train_data["Marca_Modelo"] = train_combo.where(train_combo.isin(frequent_combos), "other")
    val_data["Marca_Modelo"] = val_combo.where(val_combo.isin(frequent_combos), "other")

    return train_data, val_data


def add_premium_brand_feature(df, premium_brands=DEFAULT_PREMIUM_BRANDS,
                              brand_col="Marca"):
    """
    Adds a binary indicator for premium or high-end brands.

    Arguments:
        df (pd.DataFrame): dataset containing the brand column
        premium_brands (list[str]): brand names treated as high-end
        brand_col (str): brand column

    Returns:
        pd.DataFrame: dataset with the premium-brand indicator
    """
    data = df.copy()
    data["Es_Alta_Gama"] = data[brand_col].isin(premium_brands).astype(int)

    return data


def add_cilindrada_missing_indicator(train_df, val_df,
                                     cilindrada_col="Cilindrada"):
    """
    Adds a missingness indicator for engine displacement and imputes missing values.

    The imputation value is computed from the training set only, so validation
    does not leak information into preprocessing.

    Arguments:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        cilindrada_col (str): engine displacement column

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and validation datasets with
        the missingness indicator and imputed displacement
    """
    train_data = train_df.copy()
    val_data = val_df.copy()

    train_data[cilindrada_col] = pd.to_numeric(train_data[cilindrada_col], errors="coerce")
    val_data[cilindrada_col] = pd.to_numeric(val_data[cilindrada_col], errors="coerce")

    train_data[f"{cilindrada_col}_missing"] = train_data[cilindrada_col].isna().astype(int)
    val_data[f"{cilindrada_col}_missing"] = val_data[cilindrada_col].isna().astype(int)

    train_median = train_data[cilindrada_col].median()
    train_data[cilindrada_col] = train_data[cilindrada_col].fillna(train_median)
    val_data[cilindrada_col] = val_data[cilindrada_col].fillna(train_median)

    return train_data, val_data


def add_initial_features(train_df, val_df, reference_year=DEFAULT_REFERENCE_YEAR,
                         zero_km_threshold=DEFAULT_ZERO_KM_THRESHOLD,
                         premium_brands=DEFAULT_PREMIUM_BRANDS,
                         brand_model_min_count=20):
    """
    Adds the first feature-engineering candidates used in the project.

    The function keeps train and validation transformations aligned. Frequency
    decisions for `Marca_Modelo` and imputation values for `Cilindrada` are
    learned only from the training set.

    Arguments:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        reference_year (int): year used to compute vehicle age
        zero_km_threshold (int | float): maximum kilometers considered as 0km
        premium_brands (list[str]): brand names treated as high-end
        brand_model_min_count (int): minimum train frequency required to keep a
            brand-model combination

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and validation datasets with
        the engineered features added
    """
    train_data = add_usage_features(
        train_df,
        reference_year=reference_year,
        zero_km_threshold=zero_km_threshold,
    )
    val_data = add_usage_features(
        val_df,
        reference_year=reference_year,
        zero_km_threshold=zero_km_threshold,
    )

    train_data, val_data = add_brand_model_feature(
        train_data,
        val_data,
        min_count=brand_model_min_count,
    )

    train_data = add_premium_brand_feature(train_data, premium_brands=premium_brands)
    val_data = add_premium_brand_feature(val_data, premium_brands=premium_brands)

    train_data, val_data = add_cilindrada_missing_indicator(train_data, val_data)

    return train_data, val_data


# ======================== Feature Variant Evaluation ========================

def resolve_feature_blocks(feature_blocks):
    """
    Normalizes the list of feature blocks requested for a model variant.

    Arguments:
        feature_blocks (str | list[str] | tuple[str] | None): selected feature
            blocks. Use "all" to include every available feature block

    Returns:
        list[str]: normalized feature block names
    """
    if feature_blocks is None:
        return []

    if feature_blocks == "all":
        return ALL_FEATURE_BLOCKS.copy()

    if isinstance(feature_blocks, str):
        feature_blocks = [feature_blocks]

    invalid_blocks = [
        block for block in feature_blocks
        if block not in ALL_FEATURE_BLOCKS
    ]

    if invalid_blocks:
        raise ValueError(f"Unknown feature blocks: {invalid_blocks}.")

    return list(feature_blocks)


def add_selected_features(train_df, val_df, feature_blocks=None,
                          reference_year=DEFAULT_REFERENCE_YEAR,
                          zero_km_threshold=DEFAULT_ZERO_KM_THRESHOLD,
                          premium_brands=DEFAULT_PREMIUM_BRANDS,
                          brand_model_min_count=20):
    """
    Adds only the feature-engineering blocks requested for one experiment.

    Arguments:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        feature_blocks (str | list[str] | None): blocks to add. Available blocks
            are "usage", "brand_model", "premium" and "cilindrada_missing".
            Use "all" to include every block
        reference_year (int): year used to compute vehicle age
        zero_km_threshold (int | float): maximum kilometers considered as 0km
        premium_brands (list[str]): brand names treated as high-end
        brand_model_min_count (int): minimum train frequency required to keep a
            brand-model combination

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and validation datasets with
        the selected features added
    """
    blocks = resolve_feature_blocks(feature_blocks)
    train_data = train_df.copy()
    val_data = val_df.copy()

    if "usage" in blocks:
        train_data = add_usage_features(
            train_data,
            reference_year=reference_year,
            zero_km_threshold=zero_km_threshold,
        )
        val_data = add_usage_features(
            val_data,
            reference_year=reference_year,
            zero_km_threshold=zero_km_threshold,
        )

    if "brand_model" in blocks:
        train_data, val_data = add_brand_model_feature(
            train_data,
            val_data,
            min_count=brand_model_min_count,
        )

    if "premium" in blocks:
        train_data = add_premium_brand_feature(
            train_data,
            premium_brands=premium_brands,
        )
        val_data = add_premium_brand_feature(
            val_data,
            premium_brands=premium_brands,
        )

    if "cilindrada_missing" in blocks:
        train_data, val_data = add_cilindrada_missing_indicator(
            train_data,
            val_data,
        )

    return train_data, val_data


def build_feature_variant(train_df, val_df, feature_blocks=None, drop_cols=None,
                          columns_to_drop=None, categorical_cols=None,
                          binary_missing_cols=None,
                          reference_year=DEFAULT_REFERENCE_YEAR,
                          zero_km_threshold=DEFAULT_ZERO_KM_THRESHOLD,
                          premium_brands=DEFAULT_PREMIUM_BRANDS,
                          brand_model_min_count=20):
    """
    Builds an encoded train/validation pair for one feature variant.

    Arguments:
        train_df (pd.DataFrame): training dataset before one-hot encoding
        val_df (pd.DataFrame): validation dataset before one-hot encoding
        feature_blocks (str | list[str] | None): feature blocks to add
        drop_cols (list[str] | None): extra columns to drop before encoding
        columns_to_drop (list[str] | None): base columns to drop before encoding
        categorical_cols (list[str] | None): categorical columns to one-hot encode
        binary_missing_cols (list[str] | None): binary columns where missing
            indicators should be created
        reference_year (int): year used to compute vehicle age
        zero_km_threshold (int | float): maximum kilometers considered as 0km
        premium_brands (list[str]): brand names treated as high-end
        brand_model_min_count (int): minimum train frequency required to keep a
            brand-model combination

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: encoded train, encoded
        validation and the learned one-hot categories map
    """
    train_features, val_features = add_selected_features(
        train_df,
        val_df,
        feature_blocks=feature_blocks,
        reference_year=reference_year,
        zero_km_threshold=zero_km_threshold,
        premium_brands=premium_brands,
        brand_model_min_count=brand_model_min_count,
    )

    columns_to_drop = list(columns_to_drop or DEFAULT_COLUMNS_TO_DROP)
    columns_to_drop = columns_to_drop + list(drop_cols or [])
    categorical_cols = list(categorical_cols or DEFAULT_CATEGORICAL_COLS)
    binary_missing_cols = list(binary_missing_cols or DEFAULT_BINARY_MISSING_COLS)

    train_features = prep.drop_irrelevant_columns(train_features, columns_to_drop)
    val_features = prep.drop_irrelevant_columns(val_features, columns_to_drop)

    available_categorical_cols = [
        column for column in categorical_cols
        if column in train_features.columns and column in val_features.columns
    ]
    available_binary_missing_cols = [
        column for column in binary_missing_cols
        if column in train_features.columns and column in val_features.columns
    ]

    train_encoded, categories_map = prep.one_hot_encoding(
        train_features,
        categorical_cols=available_categorical_cols,
        train=True,
        binary_missing_cols=available_binary_missing_cols,
    )

    val_encoded = prep.one_hot_encoding(
        val_features,
        categorical_cols=available_categorical_cols,
        train=False,
        categories_map=categories_map,
        binary_missing_cols=available_binary_missing_cols,
    )

    return train_encoded, val_encoded, categories_map


def evaluate_feature_variants(train_df, val_df, y_train, y_val, variants,
                              train_func, train_kwargs=None,
                              sort_by="val_rmse", **build_kwargs):
    """
    Trains and evaluates the same model across several feature variants.

    Arguments:
        train_df (pd.DataFrame): training dataset before one-hot encoding
        val_df (pd.DataFrame): validation dataset before one-hot encoding
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        variants (list[dict]): feature variant definitions. Each dictionary can
            contain name, feature_blocks and drop_cols
        train_func (callable): model training function returning
            (model, metrics, predictions)
        train_kwargs (dict | None): keyword arguments passed to train_func
        sort_by (str): metric column used to sort the comparison table
        **build_kwargs: extra arguments passed to build_feature_variant

    Returns:
        tuple[pd.DataFrame, dict]: comparison table and detailed fitted results
        keyed by variant name
    """
    train_kwargs = train_kwargs or {}
    rows = []
    fitted_results = {}

    for variant in variants:
        variant_name = variant["name"]
        feature_blocks = variant.get("feature_blocks", [])
        drop_cols = variant.get("drop_cols", [])

        X_train_variant, X_val_variant, categories_map = build_feature_variant(
            train_df,
            val_df,
            feature_blocks=feature_blocks,
            drop_cols=drop_cols,
            **build_kwargs,
        )

        model, metrics, predictions = train_func(
            X_train_variant,
            y_train,
            X_val_variant,
            y_val,
            **train_kwargs,
        )

        row = metrics.iloc[0].to_dict()
        row.update({
            "variant": variant_name,
            "feature_blocks": ", ".join(resolve_feature_blocks(feature_blocks)) or "none",
            "dropped_cols": ", ".join(drop_cols) or "none",
            "n_features": X_train_variant.shape[1],
        })
        rows.append(row)

        fitted_results[variant_name] = {
            "model": model,
            "metrics": metrics,
            "predictions": predictions,
            "X_train": X_train_variant,
            "X_val": X_val_variant,
            "categories_map": categories_map,
        }

    comparison = pd.DataFrame(rows)
    ordered_cols = [
        "variant",
        "feature_blocks",
        "dropped_cols",
        "n_features",
        "train_rmse",
        "val_rmse",
        "train_mae",
        "val_mae",
        "train_r2",
        "val_r2",
        "train_mse",
        "val_mse",
    ]
    comparison = comparison[
        [column for column in ordered_cols if column in comparison.columns]
    ]

    if sort_by in comparison.columns:
        comparison = comparison.sort_values(sort_by).reset_index(drop=True)

    return comparison, fitted_results


def build_column_drop_variants(base_name, feature_blocks=None, columns_to_test=None,
                               include_base=True, include_all_dropped=True):
    """
    Builds variants that compare a base feature set against dropped columns.

    This is useful for feature ablation experiments where the goal is to check
    whether specific original columns, such as color or backup camera, improve
    or hurt validation performance.

    Arguments:
        base_name (str): name used for the base variant
        feature_blocks (str | list[str] | None): feature blocks used by every
            generated variant
        columns_to_test (list[str] | dict[str, str] | None): columns to drop.
            If a dictionary is provided, keys are display labels and values are
            column names
        include_base (bool): whether to include the no-drop base variant
        include_all_dropped (bool): whether to include one variant dropping all
            tested columns together

    Returns:
        list[dict]: variant definitions compatible with evaluate_feature_variants
    """
    columns_to_test = columns_to_test or []

    if isinstance(columns_to_test, dict):
        drop_items = list(columns_to_test.items())
    else:
        drop_items = [
            (_normalize_category_text(column).replace(" ", "_"), column)
            for column in columns_to_test
        ]

    variants = []

    if include_base:
        variants.append({
            "name": base_name,
            "feature_blocks": feature_blocks or [],
        })

    for label, column in drop_items:
        variants.append({
            "name": f"{base_name}_without_{label}",
            "feature_blocks": feature_blocks or [],
            "drop_cols": [column],
        })

    if include_all_dropped and len(drop_items) > 1:
        all_labels = "_and_".join(label for label, _ in drop_items)
        variants.append({
            "name": f"{base_name}_without_{all_labels}",
            "feature_blocks": feature_blocks or [],
            "drop_cols": [column for _, column in drop_items],
        })

    return variants


def evaluate_column_drop_variants(train_df, val_df, y_train, y_val, train_func,
                                  base_name, feature_blocks=None,
                                  columns_to_test=None, train_kwargs=None,
                                  sort_by="val_rmse", **build_kwargs):
    """
    Evaluates a base feature set while dropping selected columns.

    Arguments:
        train_df (pd.DataFrame): training dataset before one-hot encoding
        val_df (pd.DataFrame): validation dataset before one-hot encoding
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        train_func (callable): model training function returning
            (model, metrics, predictions)
        base_name (str): name used for the base variant
        feature_blocks (str | list[str] | None): feature blocks used by every
            generated variant
        columns_to_test (list[str] | dict[str, str] | None): columns to drop
        train_kwargs (dict | None): keyword arguments passed to train_func
        sort_by (str): metric column used to sort the comparison table
        **build_kwargs: extra arguments passed to build_feature_variant

    Returns:
        tuple[pd.DataFrame, dict]: comparison table and detailed fitted results
    """
    variants = build_column_drop_variants(
        base_name=base_name,
        feature_blocks=feature_blocks,
        columns_to_test=columns_to_test,
    )

    return evaluate_feature_variants(
        train_df,
        val_df,
        y_train,
        y_val,
        variants=variants,
        train_func=train_func,
        train_kwargs=train_kwargs,
        sort_by=sort_by,
        **build_kwargs,
    )


def display_best_variant_summary(feature_comparison, feature_variant_results):
    """
    Builds a styled summary table for the best feature variant.

    The best variant is assumed to be the first row of `feature_comparison`,
    which should already be sorted by the validation metric of interest.

    Arguments:
        feature_comparison (pd.DataFrame): comparison table returned by
            evaluate_feature_variants
        feature_variant_results (dict): fitted results returned by
            evaluate_feature_variants

    Returns:
        pd.io.formats.style.Styler: styled one-row summary table
    """
    if feature_comparison.empty:
        raise ValueError("feature_comparison is empty.")

    best_row = feature_comparison.iloc[0]
    best_variant_name = best_row["variant"]

    if best_variant_name not in feature_variant_results:
        raise ValueError(f"Variant '{best_variant_name}' is not present in feature_variant_results.")

    best_variant = feature_variant_results[best_variant_name]

    summary = pd.DataFrame([{
        "best_variant": best_variant_name,
        "feature_blocks": best_row.get("feature_blocks", "none"),
        "dropped_cols": best_row.get("dropped_cols", "none"),
        "train_shape": best_variant["X_train"].shape,
        "validation_shape": best_variant["X_val"].shape,
        "val_rmse": best_row.get("val_rmse", np.nan),
        "val_mae": best_row.get("val_mae", np.nan),
        "val_r2": best_row.get("val_r2", np.nan),
    }])

    return summary.style.hide(axis="index").format({
        "val_rmse": "{:,.2f}",
        "val_mae": "{:,.2f}",
        "val_r2": "{:.4f}",
    })


def display_feature_comparison(feature_comparison):
    """
    Builds a styled comparison table for feature-variant experiments.

    Arguments:
        feature_comparison (pd.DataFrame): comparison table returned by
            evaluate_feature_variants

    Returns:
        pd.io.formats.style.Styler: styled comparison table
    """
    metric_formats = {
        "train_rmse": "{:,.2f}",
        "val_rmse": "{:,.2f}",
        "train_mae": "{:,.2f}",
        "val_mae": "{:,.2f}",
        "train_r2": "{:.4f}",
        "val_r2": "{:.4f}",
        "train_mse": "{:,.2f}",
        "val_mse": "{:,.2f}",
    }
    available_formats = {
        column: format_value
        for column, format_value in metric_formats.items()
        if column in feature_comparison.columns
    }

    return feature_comparison.style.hide(axis="index").format(available_formats)
