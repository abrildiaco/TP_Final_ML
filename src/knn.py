import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def impute_missing_knn(df, target_col, feature_cols, n_neighbors=5,
                       round_result=False, clip_range=None, return_audit=True):
    """
    Imputes missing values in a target column using KNN.

    Arguments:
        df (pd.DataFrame): dataset to transform
        target_col (str): column with missing values to impute
        feature_cols (list[str]): numeric features used to compute similarity
        n_neighbors (int): number of neighbors
        round_result (bool): whether to round imputed values
        clip_range (tuple | None): optional minimum and maximum allowed values
        return_audit (bool): whether to return the audit table

    Returns:
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]: transformed dataset,
        optionally with audit table
    """
    data = df.copy()

    required_cols = feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")

    knn_data = data[required_cols].copy()

    for col in required_cols:
        knn_data[col] = pd.to_numeric(knn_data[col], errors="coerce")

    missing_mask = knn_data[target_col].isna()

    scaler = StandardScaler()
    knn_data[feature_cols] = scaler.fit_transform(knn_data[feature_cols])

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed = pd.DataFrame(
        imputer.fit_transform(knn_data),
        columns=knn_data.columns,
        index=data.index
    )

    imputed_values = imputed[target_col]

    if round_result:
        imputed_values = imputed_values.round()

    if clip_range is not None:
        imputed_values = imputed_values.clip(clip_range[0], clip_range[1])

    data.loc[missing_mask, target_col] = imputed_values.loc[missing_mask]

    audit_table = pd.DataFrame({
        "row_index": data.index,
        "target_col": target_col,
        "was_missing": missing_mask,
        "imputed_value": imputed_values,
        "was_filled": missing_mask,
    })

    audit_table = audit_table[audit_table["was_missing"]].reset_index(drop=True)

    print(f"Missing rows in '{target_col}': {missing_mask.sum()}")
    print(f"Filled with KNN: {missing_mask.sum()}")

    if return_audit:
        return data, audit_table

    return data