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


# ========================= Split Diagnostics =========================
def split_size_summary(X_train, X_val):
    """
    Summarizes the number and percentage of observations in each split.

    Arguments:
        X_train (pd.DataFrame | np.ndarray): training features
        X_val (pd.DataFrame | np.ndarray): validation features

    Returns:
        pd.DataFrame: split size summary
    """
    train_size = len(X_train)
    val_size = len(X_val)
    total_size = train_size + val_size

    return pd.DataFrame({
        "split": ["train", "validation"],
        "rows": [train_size, val_size],
        "percentage": [
            round(train_size / total_size * 100, 2),
            round(val_size / total_size * 100, 2),
        ],
    })
