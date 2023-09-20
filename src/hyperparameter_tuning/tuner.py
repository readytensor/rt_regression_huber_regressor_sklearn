import math
import os
from typing import Any, Callable, Dict, Union

import numpy as np
import optuna
import pandas as pd

from config import paths
from logger import get_logger
from prediction.predictor_model import evaluate_predictor_model, train_predictor_model
from utils import read_json_as_dict, save_dataframe_as_csv

HPT_RESULTS_FILE_NAME = "HPT_results.csv"

logger = get_logger(task_name="tune")
# logger = optuna.logging.get_logger(__name__)


def logger_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    """
    Logger callback for the hyperparameter tuning trials.

    Logs each trial to the logger including:
        - Iteration number
        - Current hyperparameter trial
        - Current trial objective function value
        - Best hyperparameters found so far
        - Best objective function value found so far
    """
    logger.info(f"Iteration: {trial.number}")
    logger.info(f"Current trial hyperparameters: {trial.params}")
    logger.info(f"Current trial obj func value: {trial.value}")
    logger.info(f"Best trial hyperparameters: {study.best_params}")
    logger.info(f"Best objective func value: {study.best_value}")


class HyperParameterTuner:
    """Hyperopt hyperparameter tuner class.

    Args:
        default_hps (Dict[str, Any]): Dictionary of default hyperparameter values.
        hpt_specs (Dict[str, Any]): Dictionary of hyperparameter tuning specs.
        hpt_results_dir_path (str): Dir path to save the hyperparameter tuning
            results.
        is_minimize (bool, optional): Whether the metric should be minimized.
            Defaults to True.
    """

    def __init__(
        self,
        default_hyperparameters: Dict[str, Any],
        hpt_specs: Dict[str, Any],
        hpt_results_dir_path: str,
        is_minimize: bool = True,
    ):
        """Initializes an instance of the hyperparameter tuner.

        Args:
            default_hyperparameters: Dictionary of default hyperparameter values.
            hpt_specs: Dictionary of hyperparameter tuning specs.
            hpt_results_dir_path: Dir path to save the hyperparameter tuning results.
            is_minimize:  Whether the metric should be minimized or maximized.
                Defaults to True.
        """
        self.default_hyperparameters = default_hyperparameters
        self.hpt_specs = hpt_specs
        self.hpt_results_dir_path = hpt_results_dir_path
        self.is_minimize = is_minimize
        self.num_trials = hpt_specs.get("num_trials", 20)
        assert self.num_trials >= 2, "Hyperparameter Tuning needs at least 2 trials"
        # Create optuna study
        self.study = optuna.create_study(
            study_name="optuna_hyperparameter_tuning_study",
            # always minimize because we handle the direction in the obj. function
            direction="minimize",
            # specify the sampler, with a fixed seed for reproducibility
            sampler=optuna.samplers.TPESampler(seed=5),
        )

    def run_hyperparameter_tuning(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Runs the hyperparameter tuning process.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A dictionary containing the best model name, hyperparameters, and score.
        """
        objective_func = self._get_objective_func(train_X, train_y, valid_X, valid_y)
        self.study.optimize(
            # the objective function to minimize
            func=objective_func,
            # Allow up to this many function evaluations before returning,
            n_trials=self.num_trials,
            # the number of parallel jobs, change this if you have multiple cores
            n_jobs=1,
            # Determine whether to automatically run garbage collection after
            # each trial.
            gc_after_trial=True,
            # callback
            callbacks=[logger_callback],
        )
        self.save_hpt_summary_results()
        return self.study.best_params

    def _get_objective_func(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray],
    ) -> Callable:
        """Gets the objective function for hyperparameter tuning.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A callable objective function for hyperparameter tuning.
        """

        def objective_func(trial):
            """Build a model from this hyper parameter permutation and evaluate
            its performance"""
            # extra hyperparameters from trial
            hyperparameters = self._extract_hyperparameters_from_trial(trial)
            # train model
            classifier = train_predictor_model(train_X, train_y, hyperparameters)
            # evaluate the model
            score = round(evaluate_predictor_model(classifier, valid_X, valid_y), 6)
            if np.isnan(score) or math.isinf(score):
                # sometimes loss becomes inf/na, so use a large "bad" value
                score = 1.0e6 if self.is_minimize else -1.0e6
            # If this is a maximization metric then return negative of it
            return score if self.is_minimize else -score

        return objective_func

    def _extract_hyperparameters_from_trial(self, trial: Any) -> Dict[str, Any]:
        """
        Extract the hyperparameters from trial object to pass to the model training
        function.

        Args:
            trial (Any): Dictionary containing the hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary containing the properly formatted
                            hyperparameters.
        """
        hyperparameters = {}
        trial_suggest_methods = {
            ("categorical", None): trial.suggest_categorical,
            ("int", "uniform"): trial.suggest_int,
            ("int", "log-uniform"): lambda name, low, high: trial.suggest_int(
                name, low, high, log=True
            ),
            ("real", "uniform"): trial.suggest_float,
            ("real", "log-uniform"): lambda name, low, high: trial.suggest_float(
                name, low, high, log=True
            ),
        }
        for hp_obj in self.hpt_specs["hyperparameters"]:
            method_key = (hp_obj["type"], hp_obj.get("search_type"))
            suggest_method = trial_suggest_methods.get(method_key)

            if suggest_method is None:
                raise ValueError(
                    "Error creating Hyper-Param Grid. "
                    f"Undefined value type: {hp_obj['type']} "
                    f"or search_type: {hp_obj['search_type']}. "
                    "Verify hpt_config.json file."
                )

            if hp_obj["type"] == "categorical":
                hyperparameters[hp_obj["name"]] = suggest_method(
                    hp_obj["name"], hp_obj["categories"]
                )
            else:
                hyperparameters[hp_obj["name"]] = suggest_method(
                    hp_obj["name"], hp_obj["range_low"], hp_obj["range_high"]
                )
        return hyperparameters

    def save_hpt_summary_results(self):
        """Save the hyperparameter tuning results to a file."""
        # save trial results
        hpt_results_df = self.study.trials_dataframe()
        if not os.path.exists(self.hpt_results_dir_path):
            os.makedirs(self.hpt_results_dir_path)
        save_dataframe_as_csv(
            hpt_results_df,
            os.path.join(self.hpt_results_dir_path, HPT_RESULTS_FILE_NAME),
        )


def tune_hyperparameters(
    train_X: Union[pd.DataFrame, np.ndarray],
    train_y: Union[pd.Series, np.ndarray],
    valid_X: Union[pd.DataFrame, np.ndarray],
    valid_y: Union[pd.Series, np.ndarray],
    hpt_results_dir_path: str,
    is_minimize: bool = True,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Scikit-Optimize (SKO) hyperparameter tuner.

    This function creates an instance of the SKOHyperparameterTuner with the
    provided hyperparameters and tuning specifications, then runs the hyperparameter
    tuning process and returns the best hyperparameters.

    Args:
        train_X (Union[pd.DataFrame, np.ndarray]): Training data features.
        train_y (Union[pd.Series, np.ndarray]): Training data labels.
        valid_X (Union[pd.DataFrame, np.ndarray]): Validation data features.
        valid_y (Union[pd.Series, np.ndarray]): Validation data labels.
        hpt_results_dir_path (str): Dir path to the hyperparameter tuning results file.
        is_minimize (bool, optional): Whether the metric should be minimized.
            Defaults to True.
        default_hyperparameters_file_path (str, optional): Path to the json file with
            default hyperparameter values.
            Defaults to the path defined in the paths.py file.
        hpt_specs_file_path (str, optional): Path to the json file with hyperparameter
            tuning specs.
            Defaults to the path defined in the paths.py file.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters.
    """
    default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    hyperparameter_tuner = HyperParameterTuner(
        default_hyperparameters=default_hyperparameters,
        hpt_specs=hpt_specs,
        hpt_results_dir_path=hpt_results_dir_path,
        is_minimize=is_minimize,
    )
    best_hyperparams = hyperparameter_tuner.run_hyperparameter_tuning(
        train_X, train_y, valid_X, valid_y
    )
    return best_hyperparams
