import numpy as np
import pandas as pd


# ========================= Basic Split Helpers =========================

def split_group_indices(indices, train_size = 0.80):
    """
    Splits indices from a single stratification group into train and validation indices.

    Arguments:
        indices (np.ndarray): indices belonging to one stratification group
        train_size (float): proportion assigned to the training set

    Returns:
        tuple[np.ndarray, np.ndarray]: train and validation indices for the group
    """
    train_end = int(round(len(indices) * train_size))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    return train_indices, val_indices


def safe_index(data, indices):
    """
    Selects rows from pandas or NumPy objects.

    Arguments:
        data (pd.DataFrame | pd.Series | np.ndarray): data to index
        indices (np.ndarray): row positions to select

    Returns:
        pd.DataFrame | pd.Series | np.ndarray: selected rows
    """
    if hasattr(data, "iloc"):
        return data.iloc[indices]

    return data[indices]


# ========================= Train Validation Split =========================

def train_val_split_stratified(X, y, stratify_by, train_size = 0.80, random_state = 42,):
    """
    Splits the dataset into training and validation sets using stratification.

    Arguments:
        X (pd.DataFrame | np.ndarray): feature matrix
        y (pd.Series | np.ndarray): target variable
        stratify_by (pd.Series | np.ndarray): values used to preserve proportions
        train_size (float): proportion of the dataset used for training
        random_state (int): seed used to make the split reproducible

    Returns:
        tuple: train and validation sets as (X, y) pairs
    """
    rng = np.random.default_rng(random_state)

    # Convert stratification groups to clean string values so missing values form their own group
    stratify_values = (
        pd.Series(stratify_by)
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    train_indices = []
    val_indices = []

    for group in np.unique(stratify_values):
        group_indices = np.where(stratify_values == group)[0]
        rng.shuffle(group_indices)

        group_train, group_val = split_group_indices(group_indices, train_size = train_size,)

        train_indices.extend(group_train)
        val_indices.extend(group_val)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Shuffle final splits so rows are not grouped by stratification group
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = safe_index(X, train_indices)
    y_train = safe_index(y, train_indices)

    X_val = safe_index(X, val_indices)
    y_val = safe_index(y, val_indices)

    return (X_train, y_train), (X_val, y_val)


# ========================= Optional Stratification Strategies =========================

def make_price_bins(y, n_bins=5):
    """
    Creates quantile-based price bins for stratified regression splits.

    Arguments:
        y (pd.Series | np.ndarray): target price variable
        n_bins (int): number of quantile bins to create

    Returns:
        pd.Series: price bin assigned to each observation
    """
    prices = pd.to_numeric(pd.Series(y), errors="coerce")

    return pd.qcut(
        prices,
        q=n_bins,
        labels=False,
        duplicates="drop",
    ).astype("Int64").astype(str)


def make_brand_price_groups(X, y, brand_column="Marca", n_price_bins=4, min_count=10):
    """
    Creates combined brand and price-bin stratification groups.

    Arguments:
        X (pd.DataFrame): feature matrix containing the brand column
        y (pd.Series | np.ndarray): target price variable
        brand_column (str): brand column name
        n_price_bins (int): number of price bins to create
        min_count (int): minimum group frequency before grouping as Other

    Returns:
        pd.Series: combined stratification group values
    """
    brand = X[brand_column].fillna("Unknown").astype(str)
    price_bins = make_price_bins(y, n_bins=n_price_bins)

    stratification_groups = brand + "_price_bin_" + price_bins

    # Very small groups make validation/test splits unstable
    group_counts = stratification_groups.value_counts()
    rare_groups = group_counts[group_counts < min_count].index

    return stratification_groups.where(~stratification_groups.isin(rare_groups), other = "Other")