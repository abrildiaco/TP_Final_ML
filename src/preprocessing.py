import numpy as np
import pandas as pd
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