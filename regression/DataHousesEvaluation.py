import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

from regression.DataHousesConfig import SCORERS, CV_STRATEGY, RANDOM_STATE


def run_hyperparameter_search(name, config, X_train, y_train):
    """Runs GridSearchCV or RandomizedSearchCV for a given model configuration."""
    print(f"Tuning: {name} ({config['search_type'].upper()} search)")

    if config["search_type"] == "grid":
        search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring="neg_mean_squared_error",
            cv=CV_STRATEGY,
            n_jobs=-1,
            return_train_score=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["param_grid"],
            n_iter=config.get("n_iter", 20),
            scoring="neg_mean_squared_error",
            cv=CV_STRATEGY,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )

    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.2f}")

    return search


def evaluate_best_model_cv(name, best_estimator, X_train, y_train):
    """Runs multi-metric cross-validation with RepeatedKFold on the best estimator."""
    cv_results = cross_validate(
        best_estimator,
        X_train,
        y_train,
        cv=CV_STRATEGY,
        scoring=SCORERS,
        n_jobs=-1,
        return_train_score=True,
    )
    return cv_results


def evaluate_on_test(estimator, X_test, y_test):
    """Evaluates a trained estimator on the hold-out test set."""
    y_pred = estimator.predict(X_test)
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MedAE": median_absolute_error(y_test, y_pred),
    }, y_pred
