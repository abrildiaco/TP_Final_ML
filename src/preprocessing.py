import numpy as np
import pandas as pd
import re
from eda_utils import normalize_category_text, invert_category_map

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

    normalized_column = data[column].apply(normalize_category_text)
    data[column] = normalized_column.map(inverted_map).fillna(normalized_column)

    return data


def extract_engine_liters(value, require_engine_context=False, allow_start_number=False):
    """
    Extracts engine displacement in liters from a text value.

    If require_engine_context=True, it only extracts values that appear with
    reliable engine-related patterns.
    """
    if pd.isna(value):
        return np.nan

    text = normalize_category_text(value)
    text = text.replace(",", ".")

    if require_engine_context:
        patterns = [
            r"\bmotor\s*(?:de)?\s*\d\.\d\s*l?\b",
            r"\b\d\.\d\s*(?:l|t|turbo|tsi|tdi|thp|vti|tce|hdi|tfsi|fsi)\b",
            r"\b\d\.\d(?:t)\b",
            r"\b\d{4}\s*cc\b"
        ]

        if allow_start_number:
            patterns.append(r"^\s*\d\.\d\b")

        matches = []

        for pattern in patterns:
            matches.extend(re.findall(pattern, text))

        if not matches:
            return np.nan

        text = " ".join(matches)

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


def fill_missing_engine_from_text(df, engine_col="Motor", text_cols=("Título", "Descripción", "Versión"), version_col="Versión", return_audit=True,):
    """
    Fills missing engine values using reliable engine patterns found in text columns.
    """
    data = df.copy()

    missing_mask = data[engine_col].isna()

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

    for col in text_cols:
        if col in data.columns and col != version_col:
            text_source = text_source + " " + data[col].fillna("").astype(str)

    extracted_from_text = text_source.apply(
        lambda value: extract_engine_liters(
            value,
            require_engine_context=True,
            allow_start_number=False
        )
    )

    extracted_liters = extracted_liters.fillna(extracted_from_text)

    fill_mask = missing_mask & extracted_liters.notna()

    data.loc[fill_mask, engine_col] = extracted_liters[fill_mask].astype(str)

    audit_table = pd.DataFrame({
        "row_index": data.index,
        "extracted_liters": extracted_liters,
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


# NO SE USA ESTA FUNCION
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


def map_column_values(df, column, value_map):
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
    data = df.copy()

    data[column] = (
        data[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(value_map)
    )

    return data


def one_hot_encoding(df, categorical_cols=None, train=True, categories_map=None):
    """
    Applies one-hot encoding to multiple categorical columns

    Arguments:
        df (pd.DataFrame): dataset to encode
        categorical_cols (list[str] | None): categorical columns to encode
        train (bool): whether the dataset is a training set
        categories_map (dict | None): categories learned from the training set

    Returns:
        If train=True: encoded dataframe and categories learned from train.
        If train=False: encoded dataframe using train categories.
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

        dummies = dummies.reindex(columns=expected_columns, fill_value=0)

        encoded_parts.append(dummies)

    data = data.drop(columns=categorical_cols, errors="ignore")
    data = pd.concat([data] + encoded_parts, axis=1)

    if train:
        return data, categories_map

    return data