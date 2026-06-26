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
        tuple: trained model and metrics table
    """
    target_train = np.log1p(y_train) if use_log_target else y_train

    model.fit(X_train, target_train)

    metrics = evaluate_regression_model(model, X_train, y_train, X_val, y_val,
                                        use_log_target = use_log_target,)

    return model, metrics

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
        tuple: trained model and metrics table
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