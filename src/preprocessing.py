import re # For regex operations in engine feature extraction

import numpy as np
import pandas as pd

from eda_utils import normalize_category_text, invert_category_map


# ========================= Column Selection =========================

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


# ========================= Value Filtering =========================

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

        # Convert values before filtering to avoid comparing strings with numbers
        values = pd.to_numeric(data[column], errors="coerce")
        data = data[(values >= min_value) & (values <= max_value)]

    return data


# ========================= Price Transformation =========================

def convert_peso_prices_to_usd(df, price_col="Precio", currency_col="Moneda", peso_symbol="$", exchange_rate=(895.25 + 913) / 2, copy=True):
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
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data.loc[peso_mask, price_col] = data.loc[peso_mask, price_col] / exchange_rate

    return data.drop(columns=[currency_col])


# ========================= Numeric Text Extraction =========================

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


# ========================= Categorical Mapping =========================

def apply_semantic_mapping(df, column, category_map):
    """
    Applies a manual semantic mapping to a categorical column.

    Arguments:
        df (pd.DataFrame): dataset containing the categorical column
        column (str): column to clean
        category_map (dict): dictionary with final values as keys and variants as values

    Returns:
        pd.DataFrame: dataset with the cleaned categorical column
    """
    data = df.copy()
    inverted_map = invert_category_map(category_map)

    # Normalize text before applying the manual mapping
    normalized_column = data[column].apply(normalize_category_text)
    data[column] = normalized_column.map(inverted_map).fillna(normalized_column)

    return data


def map_column_values(df, column, value_map):
    """
    Maps values from a column using a dictionary.

    Arguments:
        df (pd.DataFrame): dataset to transform
        column (str): column to map
        value_map (dict): original value to mapped value dictionary

    Returns:
        pd.DataFrame: dataset with mapped column values
    """
    data = df.copy()

    data[column] = (
        data[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(value_map)
    )

    return data


# ========================= Engine Feature Extraction =========================

def extract_engine_liters(value, require_engine_context=False, allow_start_number=False):
    """
    Extracts engine displacement in liters from a text value.

    Arguments:
        value (object): text value containing engine information
        require_engine_context (bool): whether to require engine-related patterns
        allow_start_number (bool): whether to allow a number at the beginning of the text

    Returns:
        float: engine displacement in liters or np.nan if it cannot be extracted
    """
    if pd.isna(value):
        return np.nan

    text = normalize_category_text(value)
    text = text.replace(",", ".")

    if require_engine_context:
        patterns = [
            r"\bmotor\s*(?:de)?\s*\d\.\d\s*l?\b",
            r"\b\d\.\d\s*(?:l|t|turbo|tsi|tdi|thp|vti|tce|hdi|tfsi|fsi|hibrida|hibrido|hybrid)\b",
            r"\b\d\.\d(?:t)\b",
            r"\b\d{4}\s*cc\b",
            r"\b\d\.\d\s*(?:cv|hp)\b",
        ]

        if allow_start_number:
            patterns.append(r"^\s*\d\.\d\b")

        matches = []

        # Keep only reliable fragments before extracting numbers
        for pattern in patterns:
            matches.extend(re.findall(pattern, text))

        if not matches:
            return np.nan

        text = " ".join(matches)

    # Separate numbers from letters so values like 1.6tsi can be read
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    for number_text in numbers:
        number = float(number_text)

        if 0.8 <= number <= 8.0:
            return round(number, 1)

        if 800 <= number <= 8000:
            return round(number / 1000, 1)

    return np.nan


def fill_missing_engine_from_text(df, engine_col="Motor", text_cols=("Título", "Descripción", "Versión"), version_col="Versión", 
                                  turbo_patterns=None, return_audit=True,):
    """
    Fills missing engine values using reliable engine patterns found in text columns.

    Arguments:
        df (pd.DataFrame): dataset containing engine and text columns
        engine_col (str): engine column name
        text_cols (tuple[str]): text columns used to search for engine information
        version_col (str): version column name
        turbo_patterns (list[str] | None): regex patterns that indicate turbo
        return_audit (bool): whether to return a table with the filling process

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: transformed dataset and optional audit table
    """
    data = df.copy()

    if turbo_patterns is None:
        turbo_patterns = [
            r"\bturbo\b",
            r"\btsi\b",
            r"\btdi\b",
            r"\btfsi\b",
            r"\btce\b",
            r"\bthp\b",
            r"\becoboost\b",
            r"\bt270\b",
            r"\d\.\d\s*t\b",
        ]

    missing_mask = (
        data[engine_col].isna()
        | data[engine_col].apply(
            lambda value: (
                False
                if pd.isna(value)
                else normalize_category_text(value) == "missing"
            )
        )
    )

    extracted_liters = pd.Series(np.nan, index=data.index)

    if version_col in data.columns:
        extracted_liters = data[version_col].apply(
            lambda value: extract_engine_liters(
                value,
                require_engine_context=True,
                allow_start_number=True
            )
        )

    text_source = pd.Series("", index=data.index)

    for column in text_cols:
        if column in data.columns:
            text_source = text_source + " " + data[column].fillna("").astype(str)

    extracted_from_text = text_source.apply(
        lambda value: extract_engine_liters(
            value,
            require_engine_context=True,
            allow_start_number=False
        )
    )

    extracted_liters = extracted_liters.fillna(extracted_from_text)

    extracted_has_turbo = text_source.apply(
        lambda value: has_turbo(value, turbo_patterns)
    )

    fill_mask = missing_mask & extracted_liters.notna()

    extracted_engine_text = extracted_liters.round(1).astype(str)
    extracted_engine_text = extracted_engine_text.where(
        extracted_has_turbo.eq(0),
        extracted_engine_text + " turbo"
    )

    data.loc[fill_mask, engine_col] = extracted_engine_text.loc[fill_mask]

    audit_table = pd.DataFrame({
        "row_index": data.index,
        "extracted_liters": extracted_liters,
        "extracted_has_turbo": extracted_has_turbo,
        "extracted_engine_text": extracted_engine_text,
        "was_missing": missing_mask,
        "was_filled": fill_mask,
    })

    audit_table = audit_table[audit_table["was_missing"]].reset_index(drop=True)

    print(f"Missing rows in '{engine_col}': {missing_mask.sum()}")
    print(f"Filled from text: {fill_mask.sum()}")
    print(f"Still missing after text search: {missing_mask.sum() - fill_mask.sum()}")

    if return_audit:
        return data, audit_table

    return data


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

# ========================= Fill Missing =========================
def fill_missing_from_single_text_match(df, target_col, matches_df, matched_col="matched_categories", row_index_col="row_index", separator=" | ", missing_values=("missing",)):
    """
    Fills missing target values using row-level text matches with a single detected category.

    Arguments:
        df (pd.DataFrame): dataset containing the target column
        target_col (str): column with missing values to fill
        matches_df (pd.DataFrame): row-level table containing matched categories
        matched_col (str): column with matched categories separated by a separator
        row_index_col (str): column containing original dataframe row indexes
        separator (str): separator used between multiple matched categories
        missing_values (tuple[str]): text values treated as missing

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: transformed dataset and audit table
    """
    data = df.copy()
    candidate_matches = matches_df.copy()

    normalized_missing_values = {
        normalize_category_text(value)
        for value in missing_values
    }

    missing_mask = (
        data[target_col].isna()
        | data[target_col].apply(
            lambda value: (
                False
                if pd.isna(value)
                else normalize_category_text(value) in normalized_missing_values
            )
        )
    )

    candidate_matches[matched_col] = candidate_matches[matched_col].fillna("").astype(str)

    candidate_matches["matched_list"] = candidate_matches[matched_col].apply(
        lambda value: [
            item.strip()
            for item in value.split(separator)
            if item.strip()
        ]
    )

    candidate_matches["n_unique_matches"] = candidate_matches["matched_list"].apply(
        lambda values: len(set(values))
    )

    candidate_matches["fill_value"] = candidate_matches["matched_list"].apply(
        lambda values: values[0] if len(set(values)) == 1 else np.nan
    )

    candidate_matches = candidate_matches[
        candidate_matches[row_index_col].isin(data.index)
    ].copy()

    candidate_matches["target_is_missing"] = candidate_matches[row_index_col].map(missing_mask)

    fill_candidates = candidate_matches[
        candidate_matches["target_is_missing"]
        & candidate_matches["fill_value"].notna()
    ].copy()

    fill_values = fill_candidates.set_index(row_index_col)["fill_value"]

    data.loc[fill_values.index, target_col] = fill_values

    audit_table = fill_candidates[[
        row_index_col,
        matched_col,
        "n_unique_matches",
        "fill_value",
    ]].copy()

    audit_table["target_col"] = target_col

    missing_after = (
        data[target_col].isna()
        | data[target_col].apply(
            lambda value: (
                False
                if pd.isna(value)
                else normalize_category_text(value) in normalized_missing_values
            )
        )
    )

    print(f"Missing rows in '{target_col}' before filling: {missing_mask.sum()}")
    print(f"Rows filled from text matches: {len(fill_values)}")
    print(f"Missing rows in '{target_col}' after filling: {missing_after.sum()}")

    return data, audit_table


# ========================= Encoding =========================

def one_hot_encoding(df, categorical_cols=None, train=True, categories_map=None):
    """
    Applies one-hot encoding to multiple categorical columns.

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
        categories_map = {
            column: sorted(data[column].dropna().astype(str).unique())
            for column in categorical_cols
        }

    if categories_map is None:
        raise ValueError("categories_map must be provided when train=False.")

    encoded_parts = []

    for column in categorical_cols:
        column_data = data[column].astype(str)

        dummies = pd.get_dummies(column_data, prefix=column, dtype=int)
        expected_columns = [f"{column}_{category}" for category in categories_map[column]]

        # Force validation and test to keep the same dummy columns as train
        dummies = dummies.reindex(columns=expected_columns, fill_value=0)

        encoded_parts.append(dummies)

    data = data.drop(columns=categorical_cols, errors="ignore")
    data = pd.concat([data] + encoded_parts, axis=1)

    if train:
        return data, categories_map

    return data
