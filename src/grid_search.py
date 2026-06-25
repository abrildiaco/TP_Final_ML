import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from modeling import (
    evaluate_regression_model,
    build_ridge_regression_model,
    build_lasso_regression_model,
)


# =========================  Core CV Search  =========================

def search_best_alpha_cv(model_builder, X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100), cv=5, random_state=42,):
    """
    Searches the best alpha using K-Fold Cross Validation entirely within the training set

    For each alpha, the training set is split into k folds. The model
    is trained on k-1 folds and evaluated on the remaining fold. This
    is repeated k times and the mean MSE across folds is used to rank
    alphas. The alpha with the lowest mean CV MSE is selected.

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

    for alpha in alphas:
        fold_mse_scores = []

        for train_idx, val_idx in kfold.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val   = y_train.iloc[val_idx]

            model = model_builder(alpha=alpha)
            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)
            fold_mse_scores.append(mean_squared_error(y_fold_val, predictions))

        results.append({
            "alpha": alpha,
            "mean_cv_mse": sum(fold_mse_scores) / len(fold_mse_scores),
            "std_cv_mse": pd.Series(fold_mse_scores).std(),
        })

    results = (pd.DataFrame(results).sort_values("mean_cv_mse").reset_index(drop=True))

    best_alpha = results.loc[0, "alpha"]
    return best_alpha, results


# =========================  Ridge & Lasso Alpha Search  =========================

def find_best_alpha_ridge(X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 500, 1000), cv=5, random_state=42,):
    """
    Finds the best alpha for Ridge regression using K-Fold CV on the
    training set, then retrains the final model on the full training set
    using that alpha.

    Arguments:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        alphas (tuple | list): alpha values to evaluate
        cv (int): number of folds
        random_state (int): random seed

    Returns:
        best_alpha (float): alpha with the lowest mean CV MSE
        cv_results (pd.DataFrame): CV results for each alpha
    """
    best_alpha, cv_results = search_best_alpha_cv(
        build_ridge_regression_model,
        X_train,
        y_train,
        alphas=alphas,
        cv=cv,
        random_state=random_state,
    )

    return best_alpha, cv_results


def find_best_alpha_lasso(X_train, y_train, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 500, 1000), cv=5, random_state=42,):
    """
    Finds the best alpha for Lasso regression using K-Fold CV on the
    training set, then retrains the final model on the full training set
    using that alpha.

    Arguments:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        alphas (tuple | list): alpha values to evaluate
        cv (int): number of folds
        random_state (int): random seed

    Returns:
        best_alpha (float): alpha with the lowest mean CV MSE
        cv_results (pd.DataFrame): CV results for each alpha
    """
    best_alpha, cv_results = search_best_alpha_cv(
        build_lasso_regression_model,
        X_train,
        y_train,
        alphas=alphas,
        cv=cv,
        random_state=random_state,
    )

    return best_alpha, cv_results


# =========================  Core CV Eval (hyperparams dict)  =========================

