import re  # For regex operations in engine feature extraction

import numpy as np
import pandas as pd

from eda_utils import normalize_category_text, invert_category_map


# =========================  Private Helpers  =========================

def _build_missing_mask(series, extra_missing=("missing",)):
    """
    Returns a boolean mask that is True wherever a Series value should be
    considered missing — either a real NaN or a placeholder string like
    "missing".

    Arguments:
        series (pd.Series): column to evaluate
        extra_missing (tuple[str]): additional string values treated as missing,
            compared in a case-insensitive, stripped manner

    Returns:
        pd.Series[bool]: True where the value is considered missing
    """
    nan_mask = series.isna()
    normalized = {v.strip().lower() for v in extra_missing} | {"nan"}

    # Use .where(~nan_mask) to avoid converting real NaN to the string "nan"
    # before the string comparison — otherwise isna() would miss them
    string_mask = series.where(~nan_mask).astype(str).str.strip().str.lower().isin(normalized)

    return nan_mask | string_mask


def _concat_text_columns(df, text_cols):
    """
    Horizontally concatenates several text columns into a single Series,
    joining values with a space. Columns absent from df are silently skipped,
    and NaN values are treated as empty strings so they do not propagate.

    Arguments:
        df (pd.DataFrame): dataset containing the text columns
        text_cols (tuple[str] | list[str]): columns to concatenate, in order

    Returns:
        pd.Series[str]: one combined text string per row
    """
    combined = pd.Series("", index=df.index)

    for column in text_cols:
        if column in df.columns:
            combined = combined + " " + df[column].fillna("").astype(str)

    return combined


def _log_fill_summary(target_col, missing_mask, fill_mask, source_label="text"):
    """
    Prints a short imputation summary: how many rows were missing, how many
    were filled, and how many remain unfilled after the operation.

    Arguments:
        target_col (str): name of the column being imputed (for display only)
        missing_mask (pd.Series[bool]): rows that were missing before imputation
        fill_mask (pd.Series[bool]): rows that were actually filled
        source_label (str): short label for the source used to fill values
    """
    print(f"Missing rows in '{target_col}': {missing_mask.sum()}")
    print(f"Filled from {source_label}: {fill_mask.sum()}")
    print(f"Still missing after {source_label} search: {missing_mask.sum() - fill_mask.sum()}")


def _fill_missing_from_candidates(df, target_col, fill_values, audit_data=None, missing_values=("missing"),
                                   log_label="candidates", return_audit=True,):
    """
    Fills missing values in a target column using precomputed candidate values.
    Only rows where the target is missing and the candidate value is not null are
    updated. This helper centralizes the final imputation step used by different
    text-based filling strategies.

    Arguments:
        df (pd.DataFrame): dataset containing the target column
        target_col (str): column with missing values to fill
        fill_values (pd.Series): candidate values aligned to df index
        audit_data (dict | None): additional audit columns aligned to df index
        missing_values (tuple[str]): string values treated as missing in addition to NaN
        log_label (str): label used in the printed fill summary
        return_audit (bool): whether to return the audit table alongside the dataset

    Returns:
        pd.DataFrame: imputed dataset if return_audit is False
        tuple[pd.DataFrame, pd.DataFrame]: imputed dataset and audit table if return_audit is True
    """
    data = df.copy()

    if not isinstance(fill_values, pd.Series):
        fill_values = pd.Series(fill_values, index=data.index)
    else:
        fill_values = fill_values.reindex(data.index)

    missing_mask = _build_missing_mask(data[target_col], extra_missing=missing_values)
    fill_mask = missing_mask & fill_values.notna()

    fill_candidates = fill_values.loc[fill_mask]
    has_string_candidates = fill_candidates.map(lambda value: isinstance(value, str)).any()
    target_is_string_dtype = str(data[target_col].dtype) == "string"

    if has_string_candidates or target_is_string_dtype:
        data[target_col] = data[target_col].astype("object")

    data.loc[fill_mask, target_col] = fill_values.loc[fill_mask]

    audit_table = pd.DataFrame({
        "row_index": data.index,
        "fill_value": fill_values,
        "was_missing": missing_mask,
        "was_filled": fill_mask,
    })

    if audit_data is not None:
        for column, values in audit_data.items():
            audit_table[column] = values.reindex(data.index)

    audit_table = audit_table[audit_table["was_missing"]].reset_index(drop=True)

    _log_fill_summary(target_col, missing_mask, fill_mask, source_label=log_label)

    if return_audit:
        return data, audit_table

    return data


def _single_match_fill_values(matches_df, df_index, matched_col="matched_categories",
                              row_index_col="row_index", separator=" | "):
    """
    Builds fill candidates from a row-level category match table. A value is
    accepted only when exactly one unique category was detected in the text, so
    ambiguous rows with multiple matched categories are left unfilled.

    Arguments:
        matches_df (pd.DataFrame): row-level table with detected text matches
        df_index (pd.Index): index of the original dataset
        matched_col (str): column containing matched categories joined by separator
        row_index_col (str): column with original row indexes
        separator (str): string used to split multiple matched categories

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: fill values, matched category strings,
        and number of unique matches aligned to df_index
    """
    candidate_matches = matches_df.copy()
    candidate_matches[matched_col] = candidate_matches[matched_col].fillna("").astype(str)

    candidate_matches["matched_list"] = candidate_matches[matched_col].apply(
        lambda value: [item.strip() for item in value.split(separator) if item.strip()]
    )

    candidate_matches["n_unique_matches"] = candidate_matches["matched_list"].apply(
        lambda values: len(set(values))
    )

    candidate_matches["fill_value"] = candidate_matches["matched_list"].apply(
        lambda values: values[0] if len(set(values)) == 1 else np.nan
    )

    candidate_matches = candidate_matches[
        candidate_matches[row_index_col].isin(df_index)
    ].copy()

    fill_values = pd.Series(np.nan, index=df_index, dtype="object")
    matched_values = pd.Series("", index=df_index, dtype="object")
    n_unique_matches = pd.Series(np.nan, index=df_index)

    row_indexes = candidate_matches[row_index_col]

    fill_values.loc[row_indexes] = candidate_matches["fill_value"].values
    matched_values.loc[row_indexes] = candidate_matches[matched_col].values
    n_unique_matches.loc[row_indexes] = candidate_matches["n_unique_matches"].values

    return fill_values, matched_values, n_unique_matches


# =========================  Column Selection  =========================

def drop_irrelevant_columns(df, columns_to_drop):
    """
    Removes columns that do not provide useful information for modeling.

    Arguments:
        df (pd.DataFrame): dataset to transform
        columns_to_drop (list[str]): columns to remove

    Returns:
        pd.DataFrame: dataset without the selected columns
    """
    data = df.copy()

    return data.drop(columns=columns_to_drop, errors="ignore")


def drop_rows_with_missing_values(df, column):
    """
    Removes rows where a specific column has a missing value, including
    real NaN and the placeholder string "missing".

    Arguments:
        df (pd.DataFrame): dataset to transform
        column (str): column used to identify missing rows

    Returns:
        pd.DataFrame: dataset without rows that have missing values in that column
    """
    data = df.copy()

    return data.loc[~_build_missing_mask(data[column])].reset_index(drop=True)


# =========================  Value Filtering  =========================

def remove_invalid_values(df, range_rules, copy=True):
    """
    Filters out rows whose numeric values fall outside the specified ranges.
    Non-numeric values in checked columns are coerced to NaN and dropped.

    Arguments:
        df (pd.DataFrame): dataset to filter
        range_rules (dict): column names mapped to {"min": ..., "max": ...} dicts;
            missing keys default to -inf / +inf respectively
        copy (bool): whether to return a copy instead of modifying df in place

    Returns:
        pd.DataFrame: dataset with only rows inside the specified ranges
    """
    data = df.copy() if copy else df

    for column, limits in range_rules.items():
        min_value = limits.get("min", -np.inf)
        max_value = limits.get("max", np.inf)

        # Convert values before filtering to avoid comparing strings with numbers
        values = pd.to_numeric(data[column], errors="coerce")
        data = data[(values >= min_value) & (values <= max_value)]

    return data

def detect_outliers(df, rules, mode="flag", flag_col="is_outlier"):
    """
    Detects outliers in numeric columns using fixed bounds or IQR-based bounds.
    Flag columns are added at the beginning of the dataframe for easy inspection.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        rules (dict): column names mapped to rule dicts with the following keys:
            - method (str): "fixed" or "iqr"
            - min, max (float): bounds when method is "fixed"
            - multiplier (float): IQR multiplier k, defaults to 1.5
            - allow_zero (bool): if True, zeros are never flagged and excluded
              from the IQR calculation, defaults to False
        mode (str): "flag" adds boolean outlier columns; "drop" removes outlier rows
        flag_col (str): base name for the flag columns added when mode is "flag"

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: transformed dataset and per-column summary
    """
    if mode not in ("flag", "drop"):
        raise ValueError(f"mode must be 'flag' or 'drop', got '{mode}'.")

    data = df.copy()
    outlier_mask = pd.Series(False, index=data.index)
    individual_masks = {}
    summary_rows = []

    for column, rule in rules.items():
        if column not in data.columns:
            print(f"Warning: column '{column}' not found in dataframe, skipping.")
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        method = rule.get("method")
        allow_zero = rule.get("allow_zero", False)

        if method == "fixed":
            lower = rule["min"]
            upper = rule["max"]

        elif method == "iqr":
            k = rule.get("multiplier", 1.5)
            reference_values = values[values != 0] if allow_zero else values
            q1 = reference_values.quantile(0.25)
            q3 = reference_values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr

        else:
            raise ValueError(
                f"Unknown method '{method}' for column '{column}'. "
                "Use 'fixed' or 'iqr'."
            )

        column_outlier_mask = values.isna() | (values < lower) | (values > upper)

        if allow_zero:
            column_outlier_mask = column_outlier_mask & (values != 0)

        # Collect individual masks to prepend them all at once after the loop
        individual_masks[f"{flag_col}_{column}"] = column_outlier_mask
        outlier_mask = outlier_mask | column_outlier_mask

        summary_rows.append({
            "column": column,
            "method": method,
            "allow_zero": allow_zero,
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outliers_found": int(column_outlier_mask.sum()),
            "outlier_%": round(column_outlier_mask.mean() * 100, 2),
        })

    summary = pd.DataFrame(summary_rows)

    print(f"Total rows flagged as outliers (any column): {outlier_mask.sum()} "
          f"({round(outlier_mask.mean() * 100, 2)}%)")
    print(summary.to_string(index=False))

    if mode == "flag":
        # Build flag columns first, then append the original columns after
        flag_df = pd.DataFrame({flag_col: outlier_mask, **individual_masks})
        return pd.concat([flag_df, data], axis=1)

    return data[~outlier_mask].reset_index(drop=True)

# =========================  Price Transformation  =========================

def convert_peso_prices_to_usd(df, price_col="Precio", currency_col="Moneda",
                                peso_symbol="$", exchange_rate=(895.25 + 913) / 2,
                                copy=True):
    """
    Converts prices listed in Argentine pesos to USD using a fixed exchange
    rate, then drops the currency column since all prices are unified.

    Arguments:
        df (pd.DataFrame): dataset with price and currency columns
        price_col (str): name of the price column
        currency_col (str): name of the currency column
        peso_symbol (str): value used to identify Argentine peso rows
        exchange_rate (float): ARS/USD rate used for conversion
        copy (bool): whether to return a copy instead of modifying df in place

    Returns:
        pd.DataFrame: dataset with peso prices converted to USD, currency column removed
    """
    data = df.copy() if copy else df

    peso_mask = data[currency_col].eq(peso_symbol)
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data.loc[peso_mask, price_col] = data.loc[peso_mask, price_col] / exchange_rate

    return data.drop(columns=[currency_col])


# =========================  Numeric Text Extraction  =========================

def extract_first_integer(value):
    """
    Extracts the first whitespace-separated token from a string and converts
    it to an integer. Useful for columns like "Kilómetros" that store values
    such as "150000 km" where only the number is needed.

    Arguments:
        value (object): value to transform

    Returns:
        int | float: extracted integer, or np.nan if conversion is not possible
    """
    if pd.isna(value):
        return np.nan

    first_part = str(value).strip().split()[0]
    first_part = first_part.replace(",", "")

    try:
        return int(float(first_part))
    except ValueError:
        return np.nan


# =========================  Categorical Mapping  =========================

def apply_semantic_mapping(df, column, category_map):
    """
    Standardizes a categorical column by mapping known variants to a canonical
    form. Text is normalized before matching so that differences in case,
    accents, or spacing are absorbed automatically.

    Arguments:
        df (pd.DataFrame): dataset containing the categorical column
        column (str): column to clean
        category_map (dict): canonical values as keys, lists of accepted variants as values

    Returns:
        pd.DataFrame: dataset with the categorical column standardized
    """
    data = df.copy()
    inverted_map = invert_category_map(category_map)

    # Normalize text before applying the manual mapping
    normalized_column = data[column].apply(normalize_category_text)
    data[column] = normalized_column.map(inverted_map).fillna(normalized_column)

    return data


def map_column_values(df, column, value_map):
    """
    Replaces values in a column using a lookup dictionary. Values are
    normalized to lowercase and stripped before matching, so the keys in
    value_map should follow the same convention.

    Arguments:
        df (pd.DataFrame): dataset to transform
        column (str): column whose values will be replaced
        value_map (dict): original value (lowercased, stripped) -> replacement value

    Returns:
        pd.DataFrame: dataset with the column values replaced
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


# =========================  Imputation of Values  =========================

def impute_missing_by_year(df, impute_col, year_threshold, year_col="Año", fill_value=0):
    """
    Fills missing values in a column with a default value for rows where the
    car is older than a given year threshold. Useful for any feature that did
    not exist before a certain year — if the car predates the threshold and the
    value is missing, it is safe to assume the feature is absent.

    Arguments:
        df (pd.DataFrame): dataset to transform
        impute_col (str): column with missing values to fill
        year_threshold (int): cars strictly older than this year get filled
        year_col (str): column containing the car's year, defaults to "Año"
        fill_value (int | float | str): value to assign, defaults to 0

    Returns:
        pd.DataFrame: dataset with old cars filled in impute_col

    """
    data = df.copy()

    missing_mask = _build_missing_mask(data[impute_col])
    old_car_mask = pd.to_numeric(data[year_col], errors="coerce") < year_threshold

    data.loc[missing_mask & old_car_mask, impute_col] = fill_value

    print(f"Filled {(missing_mask & old_car_mask).sum()} rows in '{impute_col}' "
          f"with {fill_value} (older than {year_threshold})")

    return data


def add_missing_indicators_for_binary_columns(df, binary_cols=None, fill_value=0):
    """
    Creates missing indicators for selected binary columns and fills the original
    missing values.

    For each selected column, the function creates a new column named
    '<column>_missing'. This indicator is 1 when the original value was missing
    and 0 otherwise. Then, the missing values in the original binary column are
    replaced with `fill_value`.

    Arguments:
        df (pd.DataFrame): dataset containing the binary columns
        binary_cols (list[str] | None): binary columns where missing indicators
            should be created
        fill_value (int | float): value used to fill missing values in the
            original binary columns

    Returns:
        pd.DataFrame: dataset with missing indicator columns added and original
            binary columns filled
    """
    data = df.copy()
    binary_cols = binary_cols or []

    for column in binary_cols:
        if column not in data.columns:
            continue

        missing_mask = _build_missing_mask(data[column])

        data[f"{column} missing"] = missing_mask.astype(int)
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[missing_mask, column] = fill_value

    return data
    
# =========================  Generic Text-Based Imputation  =========================

def fill_missing_from_text(df, target_col, text_cols, extractor,
                           extracted_col_name="extracted_value", return_audit=True):
    """
    Fills missing values by extracting candidate values directly from text columns.
    The provided extractor is applied to the concatenated text source, and only
    rows with missing target values and non-null extracted values are updated.

    Arguments:
        df (pd.DataFrame): dataset containing the target and text columns
        target_col (str): column with missing values to fill
        text_cols (tuple[str] | list[str]): text columns used as source information
        extractor (callable): function applied to the combined text to extract a value
        extracted_col_name (str): name of the extracted value column in the audit table
        return_audit (bool): whether to return the audit table alongside the dataset

    Returns:
        pd.DataFrame: imputed dataset if return_audit is False
        tuple[pd.DataFrame, pd.DataFrame]: imputed dataset and filled-row audit table
        if return_audit is True
    """
    text_source = _concat_text_columns(df, text_cols)
    extracted_values = text_source.apply(extractor)

    data, audit_table = _fill_missing_from_candidates( df, target_col=target_col, fill_values=extracted_values, 
                                                      audit_data={"text_used": text_source, extracted_col_name: extracted_values}, 
                                                      log_label="text", return_audit=True,)

    audit_table = audit_table[[
        "row_index",
        "text_used",
        extracted_col_name,
        "was_missing",
        "was_filled",
    ]]
    audit_table = audit_table[audit_table["was_filled"]].reset_index(drop=True)

    if return_audit:
        return data, audit_table

    return data

def fill_missing_from_single_text_match(df, target_col, matches_df, matched_col="matched_categories",
                                        row_index_col="row_index", separator=" | ",
                                        missing_values=("missing",)):
    """
    Fills missing values using precomputed category matches from text. A row is
    filled only when the match table contains exactly one unique detected category,
    which avoids imputing ambiguous cases.

    Arguments:
        df (pd.DataFrame): dataset containing the target column
        target_col (str): column with missing values to fill
        matches_df (pd.DataFrame): row-level table with detected categories
        matched_col (str): column in matches_df containing matched categories
        row_index_col (str): column in matches_df with original row indexes
        separator (str): string used to separate multiple matched categories
        missing_values (tuple[str]): string values treated as missing in addition to NaN

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: imputed dataset and audit table with rows
        that were actually filled
    """
    
    fill_values, matched_values, n_unique_matches = _single_match_fill_values(matches_df=matches_df, df_index=df.index, matched_col=matched_col, 
                                                                              row_index_col=row_index_col, separator=separator,)

    data, audit_table = _fill_missing_from_candidates( df, target_col=target_col, fill_values=fill_values,
                                                      audit_data={matched_col: matched_values, "n_unique_matches": n_unique_matches},
                                                        missing_values=missing_values, log_label="text matches", return_audit=True,)

    audit_table = audit_table[audit_table["was_filled"]].copy()
    audit_table = audit_table[[
        "row_index",
        matched_col,
        "n_unique_matches",
        "fill_value",
    ]]
    audit_table["target_col"] = target_col

    return data, audit_table


# =========================  Text Feature Engineering  =========================

def add_text_indicator_features(df, text_cols, terms_map, prefix="", drop_text_cols=False,
                                add_no_match_feature=False, no_match_feature_name="no_text_signal"):
    """
    Adds binary indicator features based on selected words or phrases found in
    text columns. Each key in terms_map becomes a new 0/1 column.

    Arguments:
        df (pd.DataFrame): dataset containing the text columns
        text_cols (tuple[str] | list[str]): text columns to search in
        terms_map (dict): new feature names as keys and words/phrases to search as values
        prefix (str): optional prefix added to each new feature name
        drop_text_cols (bool): whether to drop the original text columns after creating features
        add_no_match_feature (bool): whether to add a binary feature for rows
            without any detected term
        no_match_feature_name (str): name of the no-match feature before prefix

    Returns:
        pd.DataFrame: dataset with the new binary text indicator features
    """
    data = df.copy()
    text_source = _concat_text_columns(data, text_cols).apply(normalize_category_text)
    any_match = pd.Series(False, index=data.index)

    for feature_name, variants in terms_map.items():
        if isinstance(variants, str):
            variants = [variants]

        feature_values = pd.Series(False, index=data.index)

        for variant in variants:
            variant_norm = normalize_category_text(variant)
            pattern = r"\b" + re.escape(variant_norm) + r"\b"
            feature_values = feature_values | text_source.str.contains(pattern, regex=True, na=False)

        data[f"{prefix}{feature_name}"] = feature_values.astype(int)
        any_match = any_match | feature_values

    if add_no_match_feature:
        data[f"{prefix}{no_match_feature_name}"] = (~any_match).astype(int)

    if drop_text_cols:
        data = data.drop(columns=list(text_cols), errors="ignore")

    return data


def fill_missing_with_value(df, columns=None, value="missing"):
    data = df.copy()

    if columns is None:
        columns = data.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

    for column in columns:
        data[column] = data[column].astype("object")
        missing_mask = _build_missing_mask(data[column])
        data.loc[missing_mask, column] = value

    return data


# =========================  Engine Feature Extraction  =========================

def extract_engine_liters(value, require_engine_context=False, allow_start_number=False):
    """
    Extracts engine displacement in liters from a raw text value. Handles
    formats like "1.6", "1600 cc", "1.5T", "motor de 2.0 l", etc.

    Arguments:
        value (object): text value containing engine information
        require_engine_context (bool): if True, only extracts numbers that appear
            alongside engine-related keywords (e.g. "turbo", "tsi", "cc"), which
            reduces false positives in noisy text columns like titles or descriptions
        allow_start_number (bool): if require_engine_context is True, also accepts
            a bare number at the very start of the string as a valid context
            (useful for clean columns like "Version" where the engine is often first)

    Returns:
        float: engine displacement in liters, or np.nan if it cannot be extracted
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


def extract_engine_liters_from_text(value):
    """
    Extracts engine displacement from text and returns it as a string.
    If the text also mentions turbo-related terms, the returned value keeps
    that information, for example "1.6 turbo".

    Arguments:
        value (object): text value containing engine information

    Returns:
        str | float: extracted engine text, or np.nan if it cannot be extracted
    """
    liters = extract_engine_liters(value, require_engine_context=True, allow_start_number=True,)

    if pd.isna(liters):
        return np.nan

    turbo_patterns = [
        r"\bturbo\b",
        r"\bt\b",
        r"\btsi\b",
        r"\btdi\b",
        r"\bthp\b",
        r"\btce\b",
        r"\btfsi\b",
    ]

    engine_text = f"{liters:.1f}"

    if has_turbo(value, turbo_patterns):
        engine_text = f"{engine_text} turbo"

    return engine_text


def has_turbo(value, turbo_patterns):
    """
    Detects whether a text value contains indicators of turbocharging.
    Returns an integer (0/1) rather than a boolean so it can be used
    directly as a numeric feature.

    Arguments:
        value (object): text value to inspect (typically the Motor column)
        turbo_patterns (list[str]): regex patterns that signal turbo,
            e.g. [r"\\bturbo\\b", r"\\btsi\\b"]

    Returns:
        int: 1 if turbo is detected, 0 otherwise
    """
    if pd.isna(value):
        return 0

    text = normalize_category_text(value)
    pattern = "|".join(turbo_patterns)

    return int(bool(re.search(pattern, text)))


# =========================  Backup Camera Feature Extraction  =========================

def extract_backup_camera(value):
    """
    Detects mentions of a backup camera in a text value. Returns 1 when a
    camera is detected, and np.nan (not 0) when nothing is found, so that
    the absence of a mention is kept distinguishable from a confirmed "no camera"
    response.

    Arguments:
        value (object): text value to inspect

    Returns:
        int | float: 1 if a backup camera is mentioned, np.nan otherwise
    """
    if pd.isna(value):
        return np.nan

    text = normalize_category_text(value)

    # Bidirectional patterns allow "camara trasera" and "trasera camara" to both match
    positive_patterns = [
        r"\bcamara\b.{0,50}\bretroceso\b",
        r"\bretroceso\b.{0,50}\bcamara\b",
        r"\bcamara\b.{0,50}\bmarcha atras\b",
        r"\bmarcha atras\b.{0,50}\bcamara\b",
        r"\bcamara\b.{0,50}\btrasera\b",
        r"\btrasera\b.{0,50}\bcamara\b",
        r"\bcamara\b.{0,50}\bestacionamiento\b",
        r"\bestacionamiento\b.{0,50}\bcamara\b",
        r"\bcamara 360\b",
    ]

    pattern = "|".join(positive_patterns)

    return 1 if re.search(pattern, text) else np.nan


# =========================  Encoding  =========================

def one_hot_encoding(df, categorical_cols=None, train=True, categories_map=None, binary_missing_cols=None, binary_fill_value=0,):
    """
    Applies one-hot encoding to categorical columns and optionally handles
    missing values in selected binary columns.

    Categorical columns are encoded with `pd.get_dummies`. When `train=True`,
    the function learns the categories present in the training data and returns
    them in `categories_map`. When `train=False`, it uses the provided
    `categories_map` so validation or test data keeps the same dummy columns as
    training.

    For selected binary columns, the function creates a missing indicator named
    '<column>_missing' and fills the original missing values with
    `binary_fill_value`. This keeps the binary feature usable while preserving
    whether the value was originally missing.

    Arguments:
        df (pd.DataFrame): dataset to encode
        categorical_cols (list[str] | None): categorical columns to one-hot encode
        train (bool): whether to learn categories from the current dataset
        categories_map (dict | None): categories learned from the training set,
            required when train=False
        binary_missing_cols (list[str] | None): binary columns where missing
            values should be converted into explicit missing indicators
        binary_fill_value (int | float): value used to fill missing values in
            the original binary columns after creating the missing indicator

    Returns:
        tuple[pd.DataFrame, dict]: encoded dataset and learned categories map if
            train=True
        pd.DataFrame: encoded dataset if train=False
    """
    data = df.copy()
    categorical_cols = categorical_cols or []
    binary_missing_cols = binary_missing_cols or []

    data = add_missing_indicators_for_binary_columns(data, binary_cols=binary_missing_cols, fill_value=binary_fill_value,)

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
        expected_columns = [
            f"{column} {category}"
            for category in categories_map[column]
        ]

        dummies = dummies.reindex(columns=expected_columns, fill_value=0)
        encoded_parts.append(dummies)

    data = data.drop(columns=categorical_cols, errors="ignore")
    data = pd.concat([data] + encoded_parts, axis=1)

    if train:
        return data, categories_map

    return data


# =========================  Feature Engeneering  =========================
