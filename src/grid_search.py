import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

from modeling import (
    build_ridge_regression_model,
    build_lasso_regression_model,
    build_random_forest_model,
    build_xgboost_model,
    get_predictions
)

import preprocessing as prep


# =========================  Core CV Search  =========================

def _search_best_alpha_cv(model_builder, X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100), cv=5, random_state=42, use_log_target=False):
    """
    Searches the best alpha using K-Fold Cross Validation entirely within the training set

    For each alpha, the training set is split into k folds. The model
    is trained on k-1 folds and evaluated on the remaining fold. This
    is repeated k times and the mean MSE across folds is used to rank
    alphas. The alpha with the lowest mean CV MSE is selected.

    If use_log_target=True, the model is trained and evaluated on log1p(y_train).

    Arguments:
        model_builder (callable): function that builds a pipeline given alpha
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        alphas (tuple | list): alpha values to evaluate
        cv (int): number of folds
        random_state (int): random seed for reproducibility

    Returns:
        best_alpha (float): alpha with the lowest mean CV MSE
        results (pd.DataFrame): mean and std CV MSE for each alpha, sorted
            by mean_cv_mse ascending
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    target = np.log1p(y_train) if use_log_target else y_train

    for alpha in alphas:
        fold_mse_scores = []

        for train_idx, val_idx in kfold.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            model = model_builder(alpha=alpha)
            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)
            fold_mse_scores.append(mean_squared_error(y_fold_val, predictions))

        results.append({
            "alpha": alpha,
            "mean_cv_mse": sum(fold_mse_scores) / len(fold_mse_scores),
            "std_cv_mse": pd.Series(fold_mse_scores).std(),
            "use_log_target": use_log_target,
        })

    results = (pd.DataFrame(results).sort_values("mean_cv_mse").reset_index(drop=True))

    best_alpha = results.loc[0, "alpha"]
    return best_alpha, results


# =========================  Ridge & Lasso Alpha Search  =========================

def find_best_alpha_ridge(X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 500, 1000), cv=5, random_state=42, use_log_target=False):
    """ Finds the best alpha for Ridge regression using K-Fold CV on the training set """

    best_alpha, cv_results = _search_best_alpha_cv(build_ridge_regression_model, X_train, y_train, alphas=alphas, 
                                                  cv=cv, random_state=random_state, use_log_target=use_log_target)

    return best_alpha, cv_results


def find_best_alpha_lasso(X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 500, 1000), cv=5, random_state=42, use_log_target=False):
    """ Finds the best alpha for LASSO regression using K-Fold CV on the training set """

    best_alpha, cv_results = _search_best_alpha_cv(build_lasso_regression_model, X_train, y_train, alphas=alphas, cv=cv, 
                                                  random_state=random_state, use_log_target=use_log_target)

    return best_alpha, cv_results


# =========================  Core CV Eval (hyperparams dict)  =========================

def _fit_model_with_optional_log_target(model, X_train, y_train, use_log_target=False):
    """
    Fits a model using the original target or log-transformed target.

    Arguments:
        model: regression model or pipeline
        X_train (pd.DataFrame): fold training features
        y_train (pd.Series): fold training target in original scale
        use_log_target (bool): whether to train with log1p(y_train)

    Returns:
        fitted model
    """
    target_train = np.log1p(y_train) if use_log_target else y_train
    model.fit(X_train, target_train)

    return model


def _compute_cv_scores(y_true, predictions):
    """
    Computes regression metrics for one validation fold.

    Arguments:
        y_true (pd.Series): true target values in original scale
        predictions (np.ndarray): predicted values in original scale

    Returns:
        dict: MSE, RMSE, MAE and R2 for the fold
    """
    return {
        "mse": mean_squared_error(y_true, predictions),
        "rmse": root_mean_squared_error(y_true, predictions),
        "mae": mean_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
    }


def _apply_fold_one_hot_encoding(X_fold_train, X_fold_val, categorical_cols=None, binary_missing_cols=None):
    """
    Applies one-hot encoding inside a cross-validation fold.

    Categories are learned only from the fold training data and then reused to
    transform the fold validation data. This avoids letting the validation fold
    influence the dummy columns created during CV.

    Arguments:
        X_fold_train (pd.DataFrame): fold training features before one-hot
        X_fold_val (pd.DataFrame): fold validation features before one-hot
        categorical_cols (list[str] | None): categorical columns to encode
        binary_missing_cols (list[str] | None): binary columns where missing
            indicators should be created before modeling

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: encoded fold train and validation
            features
    """
    categorical_cols = categorical_cols or []
    binary_missing_cols = binary_missing_cols or []

    if not categorical_cols and not binary_missing_cols:
        return X_fold_train, X_fold_val

    X_fold_train_encoded, categories_map = prep.one_hot_encoding(X_fold_train, categorical_cols=categorical_cols,
                                                                 train=True, binary_missing_cols=binary_missing_cols)

    X_fold_val_encoded = prep.one_hot_encoding(X_fold_val, categorical_cols=categorical_cols,
                                               train=False, categories_map=categories_map, binary_missing_cols=binary_missing_cols,)

    return X_fold_train_encoded, X_fold_val_encoded


def run_or_load_model_search(model_builder, X_train, y_train, param_grid, output_path, param_types, run_search=False, 
                             cv=3, random_state=42, use_log_target=False, scoring="rmse"):
    """
    Runs or loads a hyperparameter search for a regression model.

    If run_search=True, the function evaluates every hyperparameter
    combination using K-Fold Cross Validation, saves the full results to a CSV
    file and returns the best parameter combination.

    If run_search=False, the function loads a previously saved CSV file and
    extracts the best parameter combination according to the selected scoring
    metric.

    Arguments:
        model_builder (callable): function that builds the model using keyword parameters.
        X_train (pd.DataFrame): training features.
        y_train (pd.Series): training target.
        param_grid (dict): hyperparameter grid to evaluate.
        output_path (str): path where results are saved or loaded from.
        param_types (dict): expected Python type for each hyperparameter.
        run_search (bool): whether to run the search or load saved results.
        cv (int): number of cross-validation folds.
        random_state (int): random seed for reproducibility.
        use_log_target (bool): whether to train using log1p(y_train).
        scoring (str): metric used to select the best parameters.

    Returns:
        tuple[dict, pd.DataFrame]: best parameters and full search results.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if run_search:
        best_params, results = search_best_params_cv(model_builder=model_builder, param_grid=param_grid,
                                                     X_train=X_train, y_train=y_train, cv=cv, random_state=random_state, 
                                                     use_log_target=use_log_target, scoring=scoring)

        results.to_csv(output_path, index=False)

    else:
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"No valid saved results found at {output_path}. "
                "Set run_search=True to run the search first."
            )

        results = pd.read_csv(output_path)

        score_col = f"mean_cv_{scoring}"
        ascending = scoring != "r2"

        results = (results.sort_values(score_col, ascending=ascending).reset_index(drop=True))

        best_params = {}

        for param, param_type in param_types.items():
            value = results.loc[0, param]

            if pd.isna(value):
                best_params[param] = None
            elif param_type == int:
                best_params[param] = int(value)
            elif param_type == float:
                best_params[param] = float(value)
            else:
                best_params[param] = value

    return best_params, results


def search_best_params_cv(model_builder, param_grid, X_train, y_train, cv=5, random_state=42,
                          use_log_target=False, scoring="rmse", categorical_cols=None,
                          binary_missing_cols=None):
    """
    Searches the best hyperparameter combination using K-Fold Cross Validation.

    The search is performed entirely inside the training set. For each
    parameter combination, the model is trained on k-1 folds and evaluated on
    the remaining fold. Metrics are averaged across folds and the combination
    with the best selected score is returned.

    If `use_log_target=True`, each fold model is trained on log1p(y), but
    predictions are transformed back to the original target scale before
    computing metrics. This keeps MSE, RMSE and MAE interpretable in dollars.

    Arguments:
        model_builder (callable): function that builds a model using keyword
            hyperparameters, for example build_random_forest_model
        param_grid (dict): hyperparameter grid where keys are model_builder
            argument names and values are lists of values to try
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target in original scale
        cv (int): number of folds
        random_state (int): random seed for reproducibility
        use_log_target (bool): whether to train each fold model with log1p(y)
        scoring (str): metric used to choose the best parameters. Options are
            "mse", "rmse", "mae" and "r2"
        categorical_cols (list[str] | None): categorical columns to one-hot
            encode inside each CV fold. If None, no one-hot encoding is applied
        binary_missing_cols (list[str] | None): binary columns where missing
            indicators should be created inside each CV fold

    Returns:
        tuple[dict, pd.DataFrame]: best parameters and sorted CV results
    """
    valid_scoring = {"mse", "rmse", "mae", "r2"}

    if scoring not in valid_scoring:
        raise ValueError(f"scoring must be one of {valid_scoring}")

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for train_idx, val_idx in kfold.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            X_fold_train, X_fold_val = _apply_fold_one_hot_encoding(
                X_fold_train, X_fold_val, categorical_cols=categorical_cols, binary_missing_cols=binary_missing_cols
            )

            model = model_builder(**params)
            model = _fit_model_with_optional_log_target(model, X_fold_train, y_fold_train, use_log_target=use_log_target,)

            predictions = get_predictions(model, X_fold_val, use_log_target=use_log_target,)

            fold_scores.append(_compute_cv_scores(y_fold_val, predictions))

        fold_scores = pd.DataFrame(fold_scores)

        results.append({
            "params": params,
            **params,
            "mean_cv_mse": fold_scores["mse"].mean(),
            "std_cv_mse": fold_scores["mse"].std(),
            "mean_cv_rmse": fold_scores["rmse"].mean(),
            "std_cv_rmse": fold_scores["rmse"].std(),
            "mean_cv_mae": fold_scores["mae"].mean(),
            "std_cv_mae": fold_scores["mae"].std(),
            "mean_cv_r2": fold_scores["r2"].mean(),
            "std_cv_r2": fold_scores["r2"].std(),
        })

    results = pd.DataFrame(results)
    score_col = f"mean_cv_{scoring}"
    ascending = scoring != "r2"
    results = results.sort_values(score_col, ascending=ascending).reset_index(drop=True)

    best_params = results.loc[0, "params"]

    return best_params, results


# =========================  Random Forest Hyperparameter Search  =========================

def find_best_random_forest_params(X_train, y_train, param_grid, cv=5, random_state=42, use_log_target=True, 
                                   scoring="rmse", categorical_cols=None, binary_missing_cols=None):
    """
    Finds the best Random Forest hyperparameters using K-Fold CV on the
    training set.

    The default grid is intentionally small so it can run in a reasonable time.
    If a wider search is needed, pass a custom `param_grid`.
    """
    best_params, cv_results = search_best_params_cv(build_random_forest_model, param_grid, X_train, y_train, cv=cv, 
                                                    random_state=random_state, use_log_target=use_log_target, scoring=scoring,
                                                    categorical_cols=categorical_cols, binary_missing_cols=binary_missing_cols)

    return best_params, cv_results


def run_or_load_random_forest_search(X_train, y_train, param_grid, output_path, run_search=False, cv=3, 
                                     random_state=42, use_log_target=False, scoring="rmse"):
    """
    Runs or loads the Random Forest hyperparameter search.

    If run_search=True, it evaluates all parameter combinations with cross-validation
    and saves the results to a CSV file. If run_search=False, it loads the saved
    results and returns the best parameter combination.
    """

    param_types = {
        "n_estimators": int,
        "max_depth": int,
        "min_samples_split": int,
        "min_samples_leaf": int,
        "max_features": float
    }

    return run_or_load_model_search(model_builder=build_random_forest_model, X_train=X_train, y_train=y_train, param_grid=param_grid,
                                    output_path=output_path, param_types=param_types, run_search=run_search, cv=cv,
                                    random_state=random_state, use_log_target=use_log_target, scoring=scoring)


# =========================  XGBoost Hyperparameter Search  =========================

def find_best_xgboost_params( X_train, y_train, param_grid, cv=3, random_state=42, use_log_target=False, scoring="rmse"):
    """ Finds the best XGBoost hyperparameters using K-Fold CV on the training set """

    best_params, cv_results = search_best_params_cv(build_xgboost_model, param_grid, X_train, y_train, cv=cv, random_state=random_state,
                                                    use_log_target=use_log_target, scoring=scoring)

    return best_params, cv_results


def run_or_load_xgboost_search(X_train, y_train, param_grid, output_path, run_search=False, cv=3, 
                               random_state=42, use_log_target=False, scoring="rmse"):
    """
    Runs or loads the XGBoost hyperparameter search.

    If run_search=True, it evaluates all parameter combinations with cross-validation
    and saves the results to a CSV file. If run_search=False, it loads the saved
    results and returns the best parameter combination.
    """

    param_types={
        "n_estimators": int,
        "learning_rate": float,
        "max_depth": int,
        "min_child_weight": float,
        "subsample": float,
        "colsample_bytree": float,
        "reg_alpha": float,
        "reg_lambda": float
    }

    return run_or_load_model_search(model_builder=build_xgboost_model, X_train=X_train, y_train=y_train, param_grid=param_grid,
                                    output_path=output_path, param_types=param_types, run_search=run_search, cv=cv,
                                    random_state=random_state, use_log_target=use_log_target, scoring=scoring)
    

