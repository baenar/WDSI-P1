from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

RESULTS_DIR = "regression/results/attempt1"
RANDOM_STATE = 2137

SCORERS = {
    "R2": make_scorer(r2_score),
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(mean_squared_error, greater_is_better=False),
    "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
}

CV_STRATEGY = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

MODELS_CONFIG = {
    "Ridge": {
        "estimator": Ridge(),
        "search_type": "grid",
        "param_grid": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "Lasso": {
        "estimator": Lasso(max_iter=100000),
        "search_type": "grid",
        "param_grid": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0],
        },
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "search_type": "randomized",
        "param_grid": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "n_iter": 20,
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "search_type": "randomized",
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
        },
        "n_iter": 20,
    },
}
