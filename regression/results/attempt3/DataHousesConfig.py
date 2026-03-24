from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

RESULTS_DIR = "regression/results/attempt3"
RANDOM_STATE = 2137

SCORERS = {
    "R2": make_scorer(r2_score),
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(mean_squared_error, greater_is_better=False),
    "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
}

CV_STRATEGY = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

from sklearn.compose import TransformedTargetRegressor
import numpy as np

MODELS_CONFIG = {
    "Ridge": {
        "estimator": TransformedTargetRegressor(
            regressor=Pipeline([("scaler", RobustScaler()), ("ridge", Ridge())]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "search_type": "grid",
        "param_grid": {
            "regressor__ridge__alpha": [1.0, 10.0, 100.0, 1000.0, 5000.0],
        },
    },
    "Lasso": {
        "estimator": TransformedTargetRegressor(
            regressor=Pipeline([("scaler", RobustScaler()), ("lasso", Lasso(max_iter=100000))]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "search_type": "grid",
        "param_grid": {
            "regressor__lasso__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        },
    },
    "Random Forest": {
        "estimator": TransformedTargetRegressor(
            regressor=RandomForestRegressor(random_state=RANDOM_STATE),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "search_type": "randomized",
        "param_grid": {
            "regressor__n_estimators": [50, 100, 200, 300],
            "regressor__max_depth": [None, 10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "n_iter": 20,
    },
    "Gradient Boosting": {
        "estimator": TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(random_state=RANDOM_STATE),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "search_type": "randomized",
        "param_grid": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [3, 5, 7],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__subsample": [0.8, 0.9, 1.0],
        },
        "n_iter": 20,
    },
}
