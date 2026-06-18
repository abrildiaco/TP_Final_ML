import numpy as np
import pandas as pd


def split_stratum_indices(indices, train_size = 0.80):
    """
    Splits indices from a single stratum into train and validation indices.

    Arguments:
        indices (np.ndarray): indices belonging to one stratum
        train_size (float): proportion assigned to the training set

    Returns:
        tuple[np.ndarray, np.ndarray]: train and validation indices for the stratum
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

    # Convert strata to clean string values so missing values form their own group.
    stratify_values = (
        pd.Series(stratify_by)
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    train_indices = []
    val_indices = []

    for stratum in np.unique(stratify_values):
        stratum_indices = np.where(stratify_values == stratum)[0]
        rng.shuffle(stratum_indices)

        stratum_train, stratum_val = split_stratum_indices(stratum_indices, train_size = train_size,)

        train_indices.extend(stratum_train)
        val_indices.extend(stratum_val)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Shuffle final splits so rows are not grouped by stratum.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = safe_index(X, train_indices)
    y_train = safe_index(y, train_indices)

    X_val = safe_index(X, val_indices)
    y_val = safe_index(y, val_indices)

    return (X_train, y_train), (X_val, y_val)

# VER ESTOOOOOOO
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


def make_brand_price_strata(X, y, brand_column="Marca", n_price_bins=4, min_count=10):
    """
    Creates combined brand and price-bin strata.

    Arguments:
        X (pd.DataFrame): feature matrix containing the brand column
        y (pd.Series | np.ndarray): target price variable
        brand_column (str): brand column name
        n_price_bins (int): number of price bins to create
        min_count (int): minimum stratum frequency before grouping as Other

    Returns:
        pd.Series: combined strata values
    """
    brand = X[brand_column].fillna("Unknown").astype(str)
    price_bins = make_price_bins(y, n_bins=n_price_bins)

    strata = brand + "_price_bin_" + price_bins

    # Very small strata make validation/test splits unstable.
    stratum_counts = strata.value_counts()
    rare_strata = stratum_counts[stratum_counts < min_count].index

    return strata.where(~strata.isin(rare_strata), other = "Other")