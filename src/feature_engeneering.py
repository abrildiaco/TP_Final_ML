import re

import numpy as np
import pandas as pd

from eda_utils import normalize_category_text


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

    text = normalize_category_text(value)

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
