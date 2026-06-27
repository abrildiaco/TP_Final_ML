import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# =========================  Generic functions  =========================

def build_regression_pipeline(regressor):
    """Builds a generic regression pipeline with imputation, scaling and a regressor."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def _get_predictions(model, X, use_log_target=False):
    """
    Generates predictions and returns them in the original target scale.

    If the model was trained using log(Precio), predictions are transformed back
    to dollars with expm1.
    """
    predictions = model.predict(X)

    if use_log_target:
        predictions = np.expm1(predictions)

    return predictions


def build_predictions_table(y_train, train_predictions, y_val, val_predictions):
    """
    Builds a row-level prediction table for train and validation sets.

    Arguments:
        y_train (pd.Series): training target in original scale
        train_predictions (np.ndarray): training predictions in original scale
        y_val (pd.Series): validation target in original scale
        val_predictions (np.ndarray): validation predictions in original scale

    Returns:
        pd.DataFrame: prediction table with true values, predictions and errors
    """
    train_index = y_train.index if isinstance(y_train, pd.Series) else np.arange(len(y_train))
    val_index = y_val.index if isinstance(y_val, pd.Series) else np.arange(len(y_val))

    train_results = pd.DataFrame({
        "split": "train",
        "row_index": train_index,
        "y_true": y_train,
        "y_pred": train_predictions,
    }).reset_index(drop=True)

    val_results = pd.DataFrame({
        "split": "validation",
        "row_index": val_index,
        "y_true": y_val,
        "y_pred": val_predictions,
    }).reset_index(drop=True)

    predictions = pd.concat([train_results, val_results], axis=0, ignore_index=True)
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions["abs_error"] = predictions["residual"].abs()
    predictions["signed_pct_error"] = np.where(
        predictions["y_true"] != 0,
        predictions["residual"] / predictions["y_true"] * 100,
        np.nan,
    )
    predictions["abs_pct_error"] = predictions["signed_pct_error"].abs()

    return predictions


def build_context(features, split, context_cols):
    available_cols = [column for column in context_cols if column in features.columns]
    context = features[available_cols].copy()
    context["split"] = split
    context["row_index"] = features.index
    return context


def attach_prediction_context(predictions, X_train, X_val, context_cols=None):
    """
    Adds original vehicle information to a prediction table.

    Arguments:
        predictions (pd.DataFrame): output from build_predictions_table
        X_train (pd.DataFrame): training features before one-hot encoding
        X_val (pd.DataFrame): validation features before one-hot encoding
        context_cols (list[str] | None): original columns to attach

    Returns:
        pd.DataFrame: predictions with vehicle context columns
    """
    default_context_cols = [
        "Marca",
        "Modelo",
        "Versión",
        "Año",
        "Kilómetros",
        "Cilindrada",
        "Motor",
        "Transmisión",
        "Tipo de combustible",
        "Color",
        "Tipo de vendedor",
        "Con cámara de retroceso",
    ]

    if context_cols is None:
        context_cols = default_context_cols

    context_cols = [
        column for column in context_cols
        if column in X_train.columns or column in X_val.columns
    ]

    context = pd.concat(
        [
            build_context(X_train, "train", default_context_cols),
            build_context(X_val, "validation", default_context_cols),
        ],
        axis=0,
        ignore_index=True,
    )

    predictions_with_context = predictions.merge(
        context,
        on=["split", "row_index"],
        how="left",
    )

    metric_cols = [
        column for column in predictions.columns
        if column not in ["split", "row_index"]
    ]
    ordered_cols = ["split", "row_index"] + context_cols + metric_cols

    return predictions_with_context[
        [column for column in ordered_cols if column in predictions_with_context.columns]
    ]


def top_prediction_errors(predictions, split="validation", n=20, sort_by="abs_error"):
    """
    Returns the rows where the model makes the largest prediction errors.

    Arguments:
        predictions (pd.DataFrame): prediction table, optionally with context
        split (str | None): split to inspect. Use None to inspect all rows
        n (int): number of rows to return
        sort_by (str): error column used for ranking

    Returns:
        pd.DataFrame: top error rows
    """
    if sort_by not in predictions.columns:
        raise ValueError(f"Column '{sort_by}' is not present in predictions.")

    data = predictions.copy()

    if split is not None:
        data = data[data["split"] == split]

    return data.sort_values(sort_by, ascending=False).head(n).reset_index(drop=True)


def summarize_prediction_errors(predictions, group_cols, split="validation", min_count=10):
    """
    Summarizes model errors by one or more vehicle attributes.

    Positive mean_residual means the model tends to underpredict that group.
    Negative mean_residual means it tends to overpredict that group.

    Arguments:
        predictions (pd.DataFrame): prediction table with context columns
        group_cols (str | list[str]): columns used to group rows
        split (str | None): split to inspect. Use None to inspect all rows
        min_count (int): minimum number of rows required per group

    Returns:
        pd.DataFrame: grouped error summary
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    missing_cols = [column for column in group_cols if column not in predictions.columns]

    if missing_cols:
        raise ValueError(f"Missing grouping columns: {missing_cols}.")

    data = predictions.copy()

    if split is not None:
        data = data[data["split"] == split]

    summary = (
        data
        .groupby(group_cols, dropna=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            rmse=("residual", lambda values: np.sqrt(np.mean(np.square(values)))),
            mean_residual=("residual", "mean"),
            median_y_true=("y_true", "median"),
            median_y_pred=("y_pred", "median"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
        )
        .reset_index()
    )

    summary = summary[summary["count"] >= min_count].copy()
    summary["bias_direction"] = np.where(
        summary["mean_residual"] > 0,
        "underprediction",
        "overprediction",
    )

    return summary.sort_values("mae", ascending=False).reset_index(drop=True)


def evaluate_regression_predictions(y_train, train_predictions, y_val, val_predictions):
    """
    Computes regression metrics from train and validation predictions.

    Arguments:
        y_train (pd.Series): training target in original scale
        train_predictions (np.ndarray): training predictions in original scale
        y_val (pd.Series): validation target in original scale
        val_predictions (np.ndarray): validation predictions in original scale

    Returns:
        pd.DataFrame: train and validation regression metrics
    """
    metrics = {
        "train_mse": mean_squared_error(y_train, train_predictions),
        "val_mse": mean_squared_error(y_val, val_predictions),
        "train_rmse": root_mean_squared_error(y_train, train_predictions),
        "val_rmse": root_mean_squared_error(y_val, val_predictions),
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "val_mae": mean_absolute_error(y_val, val_predictions),
        "train_r2": r2_score(y_train, train_predictions),
        "val_r2": r2_score(y_val, val_predictions),
    }

    return pd.DataFrame([metrics])


def evaluate_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=False):
    """
    Evaluates a regression model on train and validation sets.

    Metrics are always reported in the original target scale, even when the
    model was trained using log(Precio). This keeps MSE, RMSE and MAE
    interpretable in dollars.

    Arguments:
        model: trained regression model
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target in original scale
        X_val (pd.DataFrame): validation features
        y_val (pd.Series): validation target in original scale
        use_log_target (bool): whether the model was trained on log1p(y)

    Returns:
        pd.DataFrame: train and validation regression metrics
    """
    train_predictions = _get_predictions(model, X_train, use_log_target)
    val_predictions = _get_predictions(model, X_val, use_log_target)

    return evaluate_regression_predictions(y_train, train_predictions, y_val, val_predictions)


def train_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=False):
    """
    Trains and evaluates any regression model.

    If `use_log_target=True`, the model is fitted on log1p(y_train), but the
    final metrics are computed after transforming predictions back to the
    original price scale in dollars.

    Arguments:
        model: regression model or pipeline
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target in original scale
        X_val (pd.DataFrame): validation features
        y_val (pd.Series): validation target in original scale
        use_log_target (bool): whether to train using log1p(y_train)

    Returns:
        tuple: trained model, metrics table and prediction table
    """
    target_train = np.log1p(y_train) if use_log_target else y_train

    model.fit(X_train, target_train)

    train_predictions = _get_predictions(model, X_train, use_log_target)
    val_predictions = _get_predictions(model, X_val, use_log_target)

    metrics = evaluate_regression_predictions(y_train, train_predictions, y_val, val_predictions)
    predictions = build_predictions_table(y_train, train_predictions, y_val, val_predictions)

    return model, metrics, predictions

# =========================  Linear Regression  =========================

def build_linear_regression_model():
    return build_regression_pipeline(LinearRegression())


def train_linear_regression(X_train, y_train, X_val, y_val, use_log_target=False):
    model = build_linear_regression_model()
    return train_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=use_log_target)

# =========================  Ridge Regression  =========================

def build_ridge_regression_model(alpha=1.0):
    return build_regression_pipeline(Ridge(alpha=alpha))


def train_ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0, use_log_target=False):
    model = build_ridge_regression_model(alpha=alpha)
    return train_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=use_log_target)

# =========================  Lasso Regression  =========================

def build_lasso_regression_model(alpha=1.0, max_iter=50000):
    return build_regression_pipeline(Lasso(alpha=alpha, max_iter=max_iter, random_state=42))

def train_lasso_regression(X_train, y_train, X_val, y_val, alpha=1.0, use_log_target=False):
    model = build_lasso_regression_model(alpha=alpha)
    return train_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=use_log_target)

# =========================  Random Forest Regressor  =========================

def build_random_forest_model(n_estimators=300,max_depth=None,min_samples_split=2,min_samples_leaf=1,
                              max_features=1.0,random_state=42,n_jobs=-1,):
    """
    Builds a Random Forest regression pipeline.

    Random Forest does not require feature scaling.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])


def train_random_forest(X_train, y_train, X_val, y_val, use_log_target=False, **model_params):
    model = build_random_forest_model(**model_params)
    return train_regression_model(model, X_train, y_train, X_val, y_val, use_log_target=use_log_target)


# =========================  XGBoost Regressor  =========================

def build_xgboost_model(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.8, 
                        colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, objective="reg:squarederror", random_state=42, n_jobs=-1,):
    """
    Builds an XGBoost regression pipeline.

    XGBoost is a tree-based boosting model, so it does not require feature
    scaling. It is useful for this problem because it can capture nonlinear
    relationships and interactions between variables such as brand, model,
    year, kilometers and engine size.

    Arguments:
        n_estimators (int): number of boosting rounds
        learning_rate (float): step size used at each boosting iteration
        max_depth (int): maximum depth of each tree
        min_child_weight (float): minimum sum of instance weights needed in a leaf
        subsample (float): fraction of rows used by each tree
        colsample_bytree (float): fraction of columns used by each tree
        reg_alpha (float): L1 regularization strength
        reg_lambda (float): L2 regularization strength
        objective (str): XGBoost regression objective
        random_state (int): random seed for reproducibility
        n_jobs (int): number of parallel threads

    Returns:
        Pipeline: XGBoost regression pipeline
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])


def train_xgboost(X_train, y_train, X_val, y_val, use_log_target=False, **model_params):
    """
    Builds, trains and evaluates an XGBoost regressor.

    If `use_log_target=True`, the model is trained on log1p(y_train), but
    predictions are transformed back to dollars before computing the metrics.

    Arguments:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target in dollars
        X_val (pd.DataFrame): validation features
        y_val (pd.Series): validation target in dollars
        use_log_target (bool): whether to train using log1p(y_train)
        **model_params: optional parameters passed to build_xgboost_model

    Returns:
        tuple: trained model, metrics table and prediction table
    """
    model = build_xgboost_model(**model_params)

    return train_regression_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        use_log_target=use_log_target,
    )
