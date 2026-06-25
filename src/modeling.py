import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

# =========================  Generic functions  =========================

def build_regression_pipeline(regressor):
    """ Builds a generic regression pipeline with imputation, scaling and a regressor """
    
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # DESPUES BOPRRAR
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def evaluate_regression_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluates a regression model on train and validation sets.

    The main metric is MSE, as requested.
    RMSE, MAE and R2 are also included as complementary metrics.
    """
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

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


def train_regression_model(model, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates any regression model.
    """
    model.fit(X_train, y_train)
    metrics = evaluate_regression_model(model, X_train, y_train, X_val, y_val,)

    return model, metrics

# =========================  Linear Regression  =========================

def build_linear_regression_model():
    return build_regression_pipeline(LinearRegression())


def train_linear_regression(X_train, y_train, X_val, y_val):
    model = build_linear_regression_model()
    return train_regression_model(model, X_train, y_train, X_val, y_val)


# =========================  Ridge Regression  =========================

def build_ridge_regression_model(alpha=1.0):
    return build_regression_pipeline(Ridge(alpha=alpha))


def train_ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0):
    model = build_ridge_regression_model(alpha=alpha)
    return train_regression_model(model, X_train, y_train, X_val, y_val)


# =========================  Lasso Regression  =========================

def build_lasso_regression_model(alpha=1.0, max_iter=50000):
    return build_regression_pipeline(Lasso(alpha=alpha, max_iter=max_iter, random_state=42))

def train_lasso_regression(X_train, y_train, X_val, y_val, alpha=1.0):
    model = build_lasso_regression_model(alpha=alpha)
    return train_regression_model(model, X_train, y_train, X_val, y_val)


# =========================  Random Forest Regressor  =========================

def build_random_forest_model(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
):
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


def train_random_forest( X_train, y_train, X_val, y_val, **model_params,):
    """ Builds, trains and evaluates a Random Forest regressor """

    model = build_random_forest_model(**model_params)
    return train_regression_model(model, X_train, y_train, X_val, y_val,)