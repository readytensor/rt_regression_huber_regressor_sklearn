import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError

from src.prediction.predictor_model import (
    Regressor,
    evaluate_predictor_model,
    load_predictor_model,
    predict_with_model,
    save_predictor_model,
    train_predictor_model,
)


# Define the hyperparameters fixture
@pytest.fixture
def hyperparameters(default_hyperparameters):
    return default_hyperparameters


@pytest.fixture
def regressor(hyperparameters):
    """Define the regressor fixture"""
    return Regressor(**hyperparameters)


@pytest.fixture
def synthetic_data():
    """Define the synthetic dataset fixture"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    train_X, train_y = X[:80], y[:80]
    test_X, test_y = X[80:], y[80:]
    return train_X, train_y, test_X, test_y


def test_build_model(default_hyperparameters):
    """
    Test if the regressor is created with the specified hyperparameters.
    """
    modified_hyperparameters = default_hyperparameters.copy()
    for key, value in modified_hyperparameters.items():
        value_type = type(value)
        if value_type == str:
            modified_hyperparameters[key] = f"{key}_test"
        elif value_type in [int, float]:
            modified_hyperparameters[key] = 42
    new_regressor = Regressor(**modified_hyperparameters)
    for param, value in modified_hyperparameters.items():
        assert getattr(new_regressor, param) == value


def test_build_model_without_hyperparameters(default_hyperparameters):
    """
    Test if the regressor is created with default hyperparameters when
    none are provided.
    """
    default_regressor = Regressor()

    # Check if the model has default hyperparameters
    for param, value in default_hyperparameters.items():
        assert getattr(default_regressor, param) == value


def test_fit_predict_evaluate(regressor, synthetic_data):
    """
    Test if the fit method trains the model correctly and if the predict and evaluate
    methods work as expected.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    regressor.fit(train_X, train_y)
    predictions = regressor.predict(test_X)
    assert predictions.shape == test_y.shape
    assert np.array_equal(predictions, predictions.astype(float))

    score = regressor.evaluate(test_X, test_y)
    assert isinstance(score, float)
    assert score <= 1


def test_save_load(tmpdir, regressor, synthetic_data, hyperparameters):
    """
    Test if the save and load methods work correctly and if the loaded model has the
    same hyperparameters and predictions as the original.
    """

    train_X, train_y, test_X, test_y = synthetic_data
    regressor.fit(train_X, train_y)

    # Specify the file path
    model_dir_path = tmpdir.mkdir("model")

    # Save the model
    regressor.save(model_dir_path)

    # Load the model
    loaded_regressor = Regressor.load(model_dir_path)

    # Check the loaded model has the same hyperparameters as the original regressor
    for param, value in hyperparameters.items():
        assert getattr(loaded_regressor, param) == value

    # Test predictions
    predictions = loaded_regressor.predict(test_X)
    assert np.array_equal(predictions, regressor.predict(test_X))

    # Test evaluation
    score = loaded_regressor.evaluate(test_X, test_y)
    assert score == regressor.evaluate(test_X, test_y)


def test_regressor_str_representation(regressor, hyperparameters):
    """
    Test the `__str__` method of the `Regressor` class.

    The test asserts that the string representation of a `Regressor` instance is
    correctly formatted and includes the model name and the correct hyperparameters.

    Args:
        regressor (Regressor): An instance of the `Regressor` class,
            created using the `hyperparameters` fixture.
        hyperparameters (dict): A dictionary of the hyperparameters used to initialize
            the `regressor`.

    Raises:
        AssertionError: If the string representation of `regressor` does not
            match the expected format or if it does not include the correct
            hyperparameters.
    """
    regressor_str = str(regressor)

    assert regressor.model_name in regressor_str
    for param in hyperparameters.keys():
        assert param in regressor_str


def test_train_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'train_predictor_model' function returns a Regressor instance with
    correct hyperparameters.
    """
    train_X, train_y, _, _ = synthetic_data
    regressor = train_predictor_model(train_X, train_y, hyperparameters)

    assert isinstance(regressor, Regressor)
    for param, value in hyperparameters.items():
        assert getattr(regressor, param) == value


def test_predict_with_model(synthetic_data, hyperparameters):
    """
    Test that the 'predict_with_model' function returns predictions of correct size
    and type.
    """
    train_X, train_y, test_X, _ = synthetic_data
    regressor = train_predictor_model(train_X, train_y, hyperparameters)
    predictions = predict_with_model(regressor, test_X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == test_X.shape[0]


def test_save_predictor_model(tmpdir, synthetic_data, hyperparameters):
    """
    Test that the 'save_predictor_model' function correctly saves a Regressor instance
    to disk.
    """
    train_X, train_y, _, _ = synthetic_data
    model_dir_path = os.path.join(tmpdir, "model")
    regressor = train_predictor_model(train_X, train_y, hyperparameters)
    save_predictor_model(regressor, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_untrained_save_predictor_model_fails(tmpdir, regressor):
    """
    Test that the 'save_predictor_model' function correctly raises  NotFittedError
    when saving an untrained regressor to disk.
    """
    with pytest.raises(NotFittedError):
        model_dir_path = os.path.join(tmpdir, "model")
        save_predictor_model(regressor, model_dir_path)


def test_load_predictor_model(tmpdir, synthetic_data, regressor, hyperparameters):
    """
    Test that the 'load_predictor_model' function correctly loads a Regressor
    instance from disk and that the loaded instance has the correct hyperparameters.
    """
    train_X, train_y, _, _ = synthetic_data
    regressor = train_predictor_model(train_X, train_y, hyperparameters)

    model_dir_path = os.path.join(tmpdir, "model")
    save_predictor_model(regressor, model_dir_path)

    loaded_clf = load_predictor_model(model_dir_path)
    assert isinstance(loaded_clf, Regressor)
    for param, value in hyperparameters.items():
        assert getattr(loaded_clf, param) == value


def test_evaluate_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'evaluate_predictor_model' function returns an score of
    correct type and within valid range.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    regressor = train_predictor_model(train_X, train_y, hyperparameters)
    score = evaluate_predictor_model(regressor, test_X, test_y)

    assert isinstance(score, float)
    assert score <= 1  # can be < 0 since we are generating random data
