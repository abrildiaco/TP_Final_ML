import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import feature_engineering as fe
import preprocessing as prep
import constants as const


RANDOM_STATE = 42
TARGET = "Precio"
BASE_CATEGORICAL_COLS = [
    "Marca",
    "Modelo",
    "Marca_Modelo",
    "Color",
    "Tipo de vendedor",
    "Tipo de combustible",
    "Transmisión",
]
TEXT_COLS_TO_DROP = ["Título", "Descripción", "Versión"]
BINARY_MISSING_COLS = ["Con cámara de retroceso"]
XGB_BASE_PARAMS = const.XGB_BASE_PARAMS
XGB_REGULARIZED_PARAMS = const.XGB_REGULARIZED_PARAMS
EXPERIMENT_METRIC_FORMAT = const.EXPERIMENT_METRIC_FORMAT
EXPERIMENT_DISPLAY_COLS = const.EXPERIMENT_DISPLAY_COLS


def deduplicate_dataset(data):
    """
    Removes exact duplicated rows before splitting the dataset.

    Arguments:
        data (pd.DataFrame): preprocessed dataset

    Returns:
        pd.DataFrame: deduplicated dataset with a clean integer index
    """
    return data.drop_duplicates().reset_index(drop=True)


def build_deduplication_summary(data, deduplicated_data):
    """
    Builds a compact before/after duplicate summary.

    Arguments:
        data (pd.DataFrame): dataset before deduplication
        deduplicated_data (pd.DataFrame): dataset after deduplication

    Returns:
        pd.DataFrame: row counts and exact duplicate counts by stage
    """
    return pd.DataFrame([
        {
            "stage": "before_deduplication",
            "rows": len(data),
            "exact_duplicates": data.duplicated().sum(),
        },
        {
            "stage": "after_deduplication",
            "rows": len(deduplicated_data),
            "exact_duplicates": deduplicated_data.duplicated().sum(),
        },
    ])


def split_after_deduplication(data, target_col=TARGET, test_size=0.2,
                              random_state=RANDOM_STATE, stratify_col="Marca"):
    """
    Splits a deduplicated dataset into train and validation sets.

    The split is stratified by brand when possible to keep a similar brand mix
    in both sets without using the target variable for stratification.

    Arguments:
        data (pd.DataFrame): deduplicated dataset
        target_col (str): target column name
        test_size (float): validation proportion
        random_state (int): random seed
        stratify_col (str): feature used for stratification

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train,
        X_val, y_train and y_val
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    stratify_values = None

    if stratify_col in X.columns:
        counts = X[stratify_col].value_counts()
        stratify_values = X[stratify_col].where(
            X[stratify_col].map(counts) >= 2,
            "other",
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def build_split_summary(X_train, X_val, y_train, y_val):
    """
    Summarizes the train-validation split and target distribution.

    Arguments:
        X_train (pd.DataFrame): training features
        X_val (pd.DataFrame): validation features
        y_train (pd.Series): training target
        y_val (pd.Series): validation target

    Returns:
        pd.DataFrame: split sizes and target summary statistics
    """
    return pd.DataFrame([
        {
            "split": "train",
            "rows": len(X_train),
            "target_median": y_train.median(),
            "target_mean": y_train.mean(),
        },
        {
            "split": "validation",
            "rows": len(X_val),
            "target_median": y_val.median(),
            "target_mean": y_val.mean(),
        },
    ])


def _available_columns(train_features, val_features, columns):
    return [
        column for column in columns
        if column in train_features.columns and column in val_features.columns
    ]


def _prepare_base_features(X_train, X_val, feature_blocks=None, drop_cols=None,
                           brand_model_min_count=20):
    train_features, val_features = fe.add_selected_features(
        X_train,
        X_val,
        feature_blocks=feature_blocks or [],
        reference_year=2025,
        zero_km_threshold=100,
        premium_brands=const.PREMIUM_BRANDS,
        brand_model_min_count=brand_model_min_count,
    )

    columns_to_drop = TEXT_COLS_TO_DROP + list(drop_cols or [])

    train_features = prep.drop_irrelevant_columns(train_features, columns_to_drop)
    val_features = prep.drop_irrelevant_columns(val_features, columns_to_drop)

    return train_features, val_features


def apply_rare_category_grouping(train_features, val_features, categorical_cols,
                                 rare_min_count=None):
    """
    Groups rare categories using frequencies learned only from train.

    Arguments:
        train_features (pd.DataFrame): training features
        val_features (pd.DataFrame): validation features
        categorical_cols (list[str]): categorical columns to group
        rare_min_count (int | None): minimum train frequency required to keep a
            category

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: transformed train and validation
        features
    """
    if rare_min_count is None:
        return train_features, val_features

    train_features, frequent_categories = prep.group_rare_categories(
        train_features,
        categorical_cols,
        min_count=rare_min_count,
        train=True,
    )

    val_features = prep.group_rare_categories(
        val_features,
        categorical_cols,
        min_count=rare_min_count,
        categories_map=frequent_categories,
        train=False,
    )

    return train_features, val_features


def build_one_hot_experiment_features(X_train, X_val, feature_blocks=None,
                                      rare_min_count=None, drop_cols=None,
                                      brand_model_min_count=20):
    """
    Builds aligned one-hot encoded train and validation features.

    Rare-category grouping and one-hot columns are learned from train only and
    then applied to validation.

    Arguments:
        X_train (pd.DataFrame): raw training features
        X_val (pd.DataFrame): raw validation features
        feature_blocks (list[str] | str | None): feature engineering blocks
        rare_min_count (int | None): minimum frequency used for rare grouping
        drop_cols (list[str] | None): extra columns to drop
        brand_model_min_count (int): minimum frequency for Marca_Modelo

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: encoded train and validation features
    """
    train_features, val_features = _prepare_base_features(
        X_train,
        X_val,
        feature_blocks=feature_blocks,
        drop_cols=drop_cols,
        brand_model_min_count=brand_model_min_count,
    )

    categorical_cols = _available_columns(
        train_features,
        val_features,
        BASE_CATEGORICAL_COLS,
    )

    train_features, val_features = apply_rare_category_grouping(
        train_features,
        val_features,
        categorical_cols,
        rare_min_count=rare_min_count,
    )

    binary_missing_cols = _available_columns(
        train_features,
        val_features,
        BINARY_MISSING_COLS,
    )

    X_train_encoded, categories_map = prep.one_hot_encoding(
        train_features,
        categorical_cols=categorical_cols,
        train=True,
        binary_missing_cols=binary_missing_cols,
    )

    X_val_encoded = prep.one_hot_encoding(
        val_features,
        categorical_cols=categorical_cols,
        train=False,
        categories_map=categories_map,
        binary_missing_cols=binary_missing_cols,
    )

    return X_train_encoded, X_val_encoded


def train_xgboost_log_experiment(X_train, y_train, X_val, y_val, params):
    """
    Trains XGBoost on log1p(price) and evaluates predictions in USD.

    Arguments:
        X_train (pd.DataFrame): encoded training features
        y_train (pd.Series): training target in USD
        X_val (pd.DataFrame): encoded validation features
        y_val (pd.Series): validation target in USD
        params (dict): XGBoost hyperparameters

    Returns:
        tuple[Pipeline, dict, pd.DataFrame]: fitted model, metrics dictionary
        and validation predictions
    """
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", XGBRegressor(**params)),
    ])

    model.fit(X_train, np.log1p(y_train))

    train_predictions = np.expm1(model.predict(X_train))
    val_predictions = np.expm1(model.predict(X_val))

    metrics = {
        "train_rmse": root_mean_squared_error(y_train, train_predictions),
        "val_rmse": root_mean_squared_error(y_val, val_predictions),
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "val_mae": mean_absolute_error(y_val, val_predictions),
        "train_r2": r2_score(y_train, train_predictions),
        "val_r2": r2_score(y_val, val_predictions),
        "n_features": X_train.shape[1],
    }

    predictions = pd.DataFrame({
        "split": "validation",
        "row_index": X_val.index,
        "y_true": y_val,
        "y_pred": val_predictions,
    })
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions["abs_error"] = predictions["residual"].abs()

    return model, metrics, predictions


def add_oof_target_encoding(X_train, y_train, X_val, columns, n_splits=5,
                            smoothing=20, random_state=RANDOM_STATE):
    """
    Adds leakage-safe target encoding columns.

    Train encodings are out-of-fold. Validation encodings are computed from the
    full training set only.

    Arguments:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        X_val (pd.DataFrame): validation features
        columns (list[str]): columns to target encode
        n_splits (int): number of folds for out-of-fold train encoding
        smoothing (float): smoothing strength toward global mean
        random_state (int): random seed

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and validation features with
        target-encoded columns added
    """
    train_features = X_train.copy()
    val_features = X_val.copy()
    y_log = np.log1p(y_train)
    global_mean = y_log.mean()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for column in columns:
        if column not in train_features.columns or column not in val_features.columns:
            continue

        encoded_col = f"{column}_TargetMean"
        train_encoded = pd.Series(global_mean, index=train_features.index, dtype=float)

        for fold_train_idx, fold_holdout_idx in kfold.split(train_features):
            fold_X = train_features.iloc[fold_train_idx]
            fold_y = y_log.iloc[fold_train_idx]

            fold_stats = (
                pd.DataFrame({
                    column: fold_X[column].astype(str),
                    "target": fold_y.values,
                })
                .groupby(column)["target"]
                .agg(["mean", "count"])
            )
            fold_smooth = (
                fold_stats["mean"] * fold_stats["count"] + global_mean * smoothing
            ) / (fold_stats["count"] + smoothing)

            holdout_values = train_features.iloc[fold_holdout_idx][column].astype(str)
            train_encoded.iloc[fold_holdout_idx] = (
                holdout_values.map(fold_smooth).fillna(global_mean)
            )

        full_stats = (
            pd.DataFrame({
                column: train_features[column].astype(str),
                "target": y_log.values,
            })
            .groupby(column)["target"]
            .agg(["mean", "count"])
        )
        full_smooth = (
            full_stats["mean"] * full_stats["count"] + global_mean * smoothing
        ) / (full_stats["count"] + smoothing)

        train_features[encoded_col] = train_encoded
        val_features[encoded_col] = (
            val_features[column].astype(str).map(full_smooth).fillna(global_mean)
        )

    return train_features, val_features


def build_target_encoded_experiment_features(
    X_train,
    y_train,
    X_val,
    feature_blocks=None,
    rare_min_count=20,
    target_encoded_cols=None,
):
    """
    Builds a target-encoded feature matrix without validation leakage.

    Original categorical columns are dropped after adding target-encoded
    numerical columns.

    Arguments:
        X_train (pd.DataFrame): raw training features
        y_train (pd.Series): training target
        X_val (pd.DataFrame): raw validation features
        feature_blocks (list[str] | str | None): feature engineering blocks
        rare_min_count (int | None): minimum frequency used for rare grouping
        target_encoded_cols (list[str] | None): columns to target encode

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: encoded train and validation features
    """
    train_features, val_features = _prepare_base_features(
        X_train,
        X_val,
        feature_blocks=feature_blocks,
    )

    categorical_cols = _available_columns(
        train_features,
        val_features,
        BASE_CATEGORICAL_COLS,
    )

    train_features, val_features = apply_rare_category_grouping(
        train_features,
        val_features,
        categorical_cols,
        rare_min_count=rare_min_count,
    )

    train_features, val_features = add_oof_target_encoding(
        train_features,
        y_train,
        val_features,
        columns=target_encoded_cols or ["Marca", "Modelo", "Marca_Modelo"],
    )

    train_features = prep.drop_irrelevant_columns(train_features, categorical_cols)
    val_features = prep.drop_irrelevant_columns(val_features, categorical_cols)

    binary_missing_cols = _available_columns(
        train_features,
        val_features,
        BINARY_MISSING_COLS,
    )

    if binary_missing_cols:
        train_features = prep.add_missing_indicators_for_binary_columns(
            train_features,
            binary_missing_cols,
        )
        val_features = prep.add_missing_indicators_for_binary_columns(
            val_features,
            binary_missing_cols,
        )

    return train_features, val_features


def fit_label_encoders(train_features, val_features, columns, unknown_value=-1):
    """
    Applies leakage-safe label encoding to selected categorical columns.

    The category-to-code mapping is learned only from train. Categories that
    appear in validation but not in train are assigned `unknown_value`.

    Arguments:
        train_features (pd.DataFrame): training features
        val_features (pd.DataFrame): validation features
        columns (list[str]): categorical columns to label encode
        unknown_value (int): code assigned to unseen validation categories

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: encoded train and validation
        features plus learned mappings
    """
    train_encoded = train_features.copy()
    val_encoded = val_features.copy()
    encoders = {}

    for column in columns:
        if column not in train_encoded.columns or column not in val_encoded.columns:
            continue

        train_values = train_encoded[column].fillna("missing").astype(str)
        val_values = val_encoded[column].fillna("missing").astype(str)

        categories = sorted(train_values.unique())
        mapping = {category: code for code, category in enumerate(categories)}

        train_encoded[column] = train_values.map(mapping).astype(int)
        val_encoded[column] = (
            val_values.map(mapping).fillna(unknown_value).astype(int)
        )
        encoders[column] = mapping

    return train_encoded, val_encoded, encoders


def build_label_encoded_experiment_features(
    X_train,
    X_val,
    feature_blocks=None,
    rare_min_count=None,
    label_encoded_cols=None,
    one_hot_cols=None,
    drop_cols=None,
):
    """
    Builds a mixed encoding variant for XGBoost experiments.

    Selected columns use leakage-safe label encoding, while remaining
    categorical columns can still use one-hot encoding.

    Arguments:
        X_train (pd.DataFrame): raw training features
        X_val (pd.DataFrame): raw validation features
        feature_blocks (list[str] | str | None): feature engineering blocks
        rare_min_count (int | None): minimum frequency used for rare grouping
        label_encoded_cols (list[str] | None): columns to label encode
        one_hot_cols (list[str] | None): columns to one-hot encode. If None,
            remaining categorical columns are one-hot encoded
        drop_cols (list[str] | None): extra columns to drop

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: encoded train features,
        encoded validation features and learned label mappings
    """
    train_features, val_features = _prepare_base_features(
        X_train,
        X_val,
        feature_blocks=feature_blocks,
        drop_cols=drop_cols,
    )

    categorical_cols = _available_columns(
        train_features,
        val_features,
        BASE_CATEGORICAL_COLS,
    )

    train_features, val_features = apply_rare_category_grouping(
        train_features,
        val_features,
        categorical_cols,
        rare_min_count=rare_min_count,
    )

    label_encoded_cols = _available_columns(
        train_features,
        val_features,
        label_encoded_cols or [],
    )

    train_features, val_features, label_maps = fit_label_encoders(
        train_features,
        val_features,
        columns=label_encoded_cols,
    )

    if one_hot_cols is None:
        one_hot_cols = [
            column for column in categorical_cols
            if column not in label_encoded_cols
        ]
    else:
        one_hot_cols = _available_columns(train_features, val_features, one_hot_cols)

    binary_missing_cols = _available_columns(
        train_features,
        val_features,
        BINARY_MISSING_COLS,
    )

    X_train_encoded, categories_map = prep.one_hot_encoding(
        train_features,
        categorical_cols=one_hot_cols,
        train=True,
        binary_missing_cols=binary_missing_cols,
    )

    X_val_encoded = prep.one_hot_encoding(
        val_features,
        categorical_cols=one_hot_cols,
        train=False,
        categories_map=categories_map,
        binary_missing_cols=binary_missing_cols,
    )

    return X_train_encoded, X_val_encoded, label_maps


def _base_result_row(config, metrics):
    row = metrics.copy()
    row.update({
        "experiment": config["experiment"],
        "params": config["params_name"],
        "rare_min_count": config.get("rare_min_count"),
        "dropped_cols": ", ".join(config.get("drop_cols", [])) or "none",
    })
    return row


def run_one_hot_experiments(X_train, y_train, X_val, y_val, experiments):
    """
    Runs a list of one-hot encoded XGBoost experiments.

    Arguments:
        X_train (pd.DataFrame): raw training features
        y_train (pd.Series): training target
        X_val (pd.DataFrame): raw validation features
        y_val (pd.Series): validation target
        experiments (list[dict]): experiment configurations

    Returns:
        tuple[pd.DataFrame, dict]: comparison table and fitted result details
    """
    rows = []
    fitted = {}

    for config in experiments:
        X_train_exp, X_val_exp = build_one_hot_experiment_features(
            X_train,
            X_val,
            feature_blocks=config.get("feature_blocks", []),
            rare_min_count=config.get("rare_min_count"),
            drop_cols=config.get("drop_cols", []),
        )

        model, metrics, predictions = train_xgboost_log_experiment(
            X_train_exp,
            y_train,
            X_val_exp,
            y_val,
            params=config["params"],
        )

        rows.append(_base_result_row(config, metrics))
        fitted[config["experiment"]] = {
            "model": model,
            "predictions": predictions,
            "X_train": X_train_exp,
            "X_val": X_val_exp,
        }

    return build_experiment_comparison(rows), fitted


def run_target_encoding_experiments(X_train, y_train, X_val, y_val, experiments):
    """
    Runs a list of target-encoded XGBoost experiments.

    Target encodings are computed without validation leakage.

    Arguments:
        X_train (pd.DataFrame): raw training features
        y_train (pd.Series): training target
        X_val (pd.DataFrame): raw validation features
        y_val (pd.Series): validation target
        experiments (list[dict]): experiment configurations

    Returns:
        tuple[pd.DataFrame, dict]: comparison table and fitted result details
    """
    rows = []
    fitted = {}

    for config in experiments:
        X_train_exp, X_val_exp = build_target_encoded_experiment_features(
            X_train,
            y_train,
            X_val,
            feature_blocks=config.get("feature_blocks", []),
            rare_min_count=config.get("rare_min_count"),
        )

        model, metrics, predictions = train_xgboost_log_experiment(
            X_train_exp,
            y_train,
            X_val_exp,
            y_val,
            params=config["params"],
        )

        row = _base_result_row(config, metrics)
        row["dropped_cols"] = "original categorical columns"
        rows.append(row)

        fitted[config["experiment"]] = {
            "model": model,
            "predictions": predictions,
            "X_train": X_train_exp,
            "X_val": X_val_exp,
        }

    return build_experiment_comparison(rows), fitted


def run_label_encoding_experiments(X_train, y_train, X_val, y_val, experiments):
    """
    Runs a list of mixed label-encoding XGBoost experiments.

    Label mappings are learned only from train and unseen validation categories
    are assigned -1.

    Arguments:
        X_train (pd.DataFrame): raw training features
        y_train (pd.Series): training target
        X_val (pd.DataFrame): raw validation features
        y_val (pd.Series): validation target
        experiments (list[dict]): experiment configurations

    Returns:
        tuple[pd.DataFrame, dict]: comparison table and fitted result details
    """
    rows = []
    fitted = {}

    for config in experiments:
        X_train_exp, X_val_exp, label_maps = build_label_encoded_experiment_features(
            X_train,
            X_val,
            feature_blocks=config.get("feature_blocks", []),
            rare_min_count=config.get("rare_min_count"),
            label_encoded_cols=config.get("label_encoded_cols", []),
            drop_cols=config.get("drop_cols", []),
        )

        model, metrics, predictions = train_xgboost_log_experiment(
            X_train_exp,
            y_train,
            X_val_exp,
            y_val,
            params=config["params"],
        )

        row = _base_result_row(config, metrics)
        row["encoding"] = "label_encoding"
        rows.append(row)

        fitted[config["experiment"]] = {
            "model": model,
            "predictions": predictions,
            "X_train": X_train_exp,
            "X_val": X_val_exp,
            "label_maps": label_maps,
        }

    return build_experiment_comparison(rows), fitted


def build_experiment_comparison(rows):
    """
    Builds a sorted experiment comparison table.

    Arguments:
        rows (list[dict]): experiment metric rows

    Returns:
        pd.DataFrame: sorted comparison table
    """
    comparison = pd.DataFrame(rows)

    if comparison.empty:
        return comparison

    return comparison.sort_values("val_rmse").reset_index(drop=True)


def style_experiment_comparison(comparison):
    """
    Returns a formatted Styler for experiment comparison tables.

    Arguments:
        comparison (pd.DataFrame): experiment comparison table

    Returns:
        pd.io.formats.style.Styler: formatted comparison table
    """
    display_cols = [
        column for column in EXPERIMENT_DISPLAY_COLS
        if column in comparison.columns
    ]

    return (
        comparison[display_cols]
        .style
        .hide(axis="index")
        .format(EXPERIMENT_METRIC_FORMAT)
    )


def select_best_experiment(comparison, *fitted_result_dicts):
    """
    Selects the best experiment according to the first row of a sorted table.

    Arguments:
        comparison (pd.DataFrame): sorted experiment comparison table
        *fitted_result_dicts: dictionaries returned by run_*_experiments

    Returns:
        tuple[str, dict]: best experiment name and fitted result details
    """
    if comparison.empty:
        raise ValueError("comparison is empty.")

    best_name = comparison.iloc[0]["experiment"]

    for fitted_results in fitted_result_dicts:
        if best_name in fitted_results:
            return best_name, fitted_results[best_name]

    raise ValueError(f"Experiment '{best_name}' was not found in fitted results.")


def build_best_experiment_summary(comparison, best_name, best_experiment):
    """
    Builds a one-row summary for the best experiment.

    Arguments:
        comparison (pd.DataFrame): sorted comparison table
        best_name (str): best experiment name
        best_experiment (dict): fitted result details

    Returns:
        pd.DataFrame: best experiment summary
    """
    best_row = comparison[comparison["experiment"] == best_name].iloc[0]

    return pd.DataFrame([{
        "best_experiment": best_name,
        "train_shape": best_experiment["X_train"].shape,
        "validation_shape": best_experiment["X_val"].shape,
        "val_rmse": best_row["val_rmse"],
        "val_mae": best_row["val_mae"],
        "val_r2": best_row["val_r2"],
    }])


def attach_validation_context(predictions, X_val):
    """
    Adds raw validation features to a validation prediction table.

    Arguments:
        predictions (pd.DataFrame): validation predictions with row_index
        X_val (pd.DataFrame): raw validation features

    Returns:
        pd.DataFrame: predictions with vehicle context columns attached
    """
    return predictions.merge(
        X_val.reset_index().rename(columns={"index": "row_index"}),
        on="row_index",
        how="left",
    )
