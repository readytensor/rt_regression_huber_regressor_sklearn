from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """Binarizes the target variable to 0/1 values."""

    def __init__(self, target_field: str) -> None:
        """
        Initializes a new instance of the `CustomTargetEncoder` class.

        Args:
            target_field: str
                Name of the target field.
        """
        self.target_field = target_field
        self.target_encoder = StandardScaler()

    def fit(self, data):
        """
        Fits the target encoder to the given targets

        Returns:
            self
        """
        if self.target_field not in data.columns:
            raise ValueError(
                "Target field not present in data. "
                f"Expecting target field: {self.target_field}"
                "Cannot encode target."
            )
        self.target_encoder.fit(data[[self.target_field]])
        return self

    def transform(self, data):
        """
        Transform the data.

        Args:
            data: pandas DataFrame - data to transform
        Returns:
            transformed data as a pandas Series if target is present in data, else None
        """
        if self.target_field in data.columns:
            transformed_targets = self.target_encoder.transform(
                data[[self.target_field]]
            )
        else:
            transformed_targets = None
        return transformed_targets

    def inverse_transform(self, transformed: Union[pd.DataFrame, np.ndarray]):
        """
        Inverse transform the data.

        Args:
            transformed: Union[pd.DataFrame, np.ndarray] - data to inverse-transform
        Returns:
            Inverse transformed data as a pandas series
        """
        if transformed.ndim == 1:
            arr_to_inverse_transform = transformed.reshape(-1, 1)
        else:
            arr_to_inverse_transform = transformed
        inverse_transformed = self.target_encoder.inverse_transform(
            arr_to_inverse_transform
        )
        return inverse_transformed


def get_target_encoder(data_schema: Any) -> "CustomTargetEncoder":
    """Create a TargetEncoder using the data_schema.

    Args:
        data_schema (Any): An instance of the RegressionSchema.

    Returns:
        A TargetEncoder instance.
    """
    # Create a target encoder instance
    encoder = CustomTargetEncoder(target_field=data_schema.target)
    return encoder


def train_target_encoder(
    target_encoder: CustomTargetEncoder, train_data: pd.DataFrame
) -> CustomTargetEncoder:
    """Train the target encoder using the given training data.

    Args:
        target_encoder (CustomTargetEncoder): A target encoder instance.
        train_data (pd.DataFrame): The training data as a pandas DataFrame.

    Returns:
        A fitted target encoder instance.
    """
    # Fit the target encoder on the training data
    target_encoder.fit(train_data)
    return target_encoder


def transform_targets(
    target_encoder: CustomTargetEncoder, data: Union[pd.DataFrame, np.ndarray]
) -> pd.Series:
    """Transform the target values using the fitted target encoder.

    Args:
        target_encoder (CustomTargetEncoder): A fitted target encoder instance.
        data (pd.DataFrame): The data as a pandas DataFrame.

    Returns:
        The transformed target values as a pandas Series.
    """
    # Transform the target values
    transformed_targets = target_encoder.transform(data)
    return transformed_targets


def inverse_transform_targets(
    target_encoder: CustomTargetEncoder, predictions: Union[pd.DataFrame, np.ndarray]
) -> pd.Series:
    """Transform the predictions values using the fitted target encoder.

    Args:
        target_encoder (CustomTargetEncoder): A fitted target encoder instance.
        predictions Union[pd.DataFrame, np.ndarray]: Predictions data.

    Returns:
        The transformed target values as a pandas Series.
    """
    inverse_transformed_predictions = target_encoder.inverse_transform(predictions)
    return inverse_transformed_predictions


def save_target_encoder(
    target_encoder: CustomTargetEncoder, file_path_and_name: str
) -> None:
    """Save a fitted label encoder to a file using joblib.

    Args:
        target_encoder (CustomTargetEncoder): A fitted target encoder instance.
        file_path_and_name (str): The filepath to save the LabelEncoder to.
    """
    joblib.dump(target_encoder, file_path_and_name)


def load_target_encoder(file_path_and_name: str) -> CustomTargetEncoder:
    """Load the fitted target encoder from the given path.

    Args:
        file_path_and_name: Path to the saved target encoder.

    Returns:
        Fitted target encoder.
    """
    return joblib.load(file_path_and_name)
