import numpy as np
import optuna
import polars as pl
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Report regression metrics.

    Args:
        y_true (Union[list, np.ndarray, torch.tensor]): Ground truth (correct) target values.
        y_pred (Union[list, np.ndarray, torch.tensor]): Estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    report = {}

    report['R2'] = r2_score(y_true, y_pred)
    report['MSE'] = mean_squared_error(y_true, y_pred)
    report['MAE'] = mean_absolute_error(y_true, y_pred)
    report['RMSE'] = root_mean_squared_error(y_true, y_pred)
    report['RMSLE'] = root_mean_squared_log_error(y_true, y_pred)

    return report


def fit_and_plot_regression_model(model, model_name: str, X: Union[pl.DataFrame, np.ndarray], y: np.ndarray, target_name: str = 'Calories_Expended') -> dict:
    """Fit a regression model and plot the predictions against the true values.

    Args:
        model: the model. Should implement a fit method and a predict method.
        model_name (str): the name of the model. Used for printing and plotting.
        X (Union[pl.DataFrame, np.ndarray]): input features.
        y (np.ndarray): target vector. Should be a 1D array.
        target_name (str, optional): the name of te target. Used for plotting. Defaults to 'Calories_Expended'.

    Returns:
        dict: a dictionary containing regression metrics.   
    """
    print('-'*50)
    print(f"Cross-validating and making predictions with {model_name}...")

    start = time.time()
    y_train_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

    print(f'Time taken: {round(time.time() - start, 2)} seconds')
    report = regression_report(y, y_train_pred)

    for metric_name, metric_value in report.items():
        print(f"{metric_name}: {round(metric_value, 5)}")
    
    # plot the predictions
    plt.figure(figsize=(8, 6))
    sns.jointplot(x=y, y=y_train_pred, kind='hist')
    plt.xlabel(f'True {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.suptitle(f'{model_name} {target_name} Predictions vs True Values', y=1.02)
    plt.show()

    return report


def objective_xgb_regressor_rmsle(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray, n_jobs: int = -1, cv: int = 10) -> float:
    """Objective function for Optuna to optimize the hyperparameters of the XGBRegressor model using RMSLE as the metric.

    Args:
        trial (optuna.trial.Trial): Optuna trial object
        X (np.ndarray): input features
        y (np.ndarray): target vector
        n_jobs (int, optional): number of cpu cores to use. Defaults to -1 (all cores).
        cv (int, optional): number of k-fold cross-validation folds. Defaults to 10.

    Returns:
        float: the root mean squared log error of the model
    """
    # define the hyperparameters to tune
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10)
    }
    # ^^ no idea if these values are optimal

    # instantiate the model
    model = XGBRegressor(**params, random_state=1738)

    # make predictions using cross-validation
    y_hat_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)

    # clip so that rsmle doesn't break
    y_hat_pred = np.clip(y_hat_pred, 1e-15, None)

    # calculate the root mean squared log error and return 
    return root_mean_squared_log_error(y, y_hat_pred)
