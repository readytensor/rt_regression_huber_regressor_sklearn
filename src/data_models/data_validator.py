import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import RegressionSchema


def get_data_validator(schema: RegressionSchema, is_train: bool) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame contains the ID field specified in the schema.
    2. That the data under ID field is unique.
    3. If `is_train` is `True`, that the input DataFrame contains the target field
        specified in the schema.
    6. That the input DataFrame contains all feature fields specified in the schema.
    7. For non-nullable features, that they do not contain null values.
    8. That numeric features do not contain non-numeric values.

    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (RegressionSchema): An instance of RegressionSchema.
        is_train (bool): Whether the data is for training or not. Determines whether
            the presence of a target field is required in the data.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):

            if schema.id not in data.columns:
                raise ValueError(
                    f"ID field '{schema.id}' is not present in the given data"
                )

            if data[schema.id].duplicated().any():
                raise ValueError(
                    f"ID field '{schema.id}' does not contain unique values"
                )

            if is_train:
                if schema.target not in data.columns:
                    raise ValueError(
                        f"Target field '{schema.target}' is not present "
                        "in the given data"
                    )

            for feature in schema.features:
                if feature not in data.columns:
                    raise ValueError(
                        f"Feature '{feature}' is not present in the given data"
                    )

            for feature in schema.non_nullable_features:
                if feature not in data.columns:
                    raise ValueError(
                        f"Feature '{feature}' is not present in the given data"
                    )

                if data[feature].isnull().any():
                    raise ValueError(
                        f"Non-nullable feature '{feature}' contains null values"
                    )

            for feature in schema.numeric_features:
                if not all(data[feature].apply(lambda x: pd.isnull(x) or np.isreal(x))):
                    raise ValueError(
                        f"Numeric feature '{feature}' contains non-numeric data"
                    )

            return data

    return DataValidator


def validate_data(
    data: pd.DataFrame, data_schema: RegressionSchema, is_train: bool
) -> pd.DataFrame:
    """
    Validates the data using the provided schema.

    Args:
        data (pd.DataFrame): The train or test data to validate.
        data_schema (RegressionSchema): An instance of
            RegressionSchema.
        is_train (bool): Whether the data is for training or not.

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_data_validator(data_schema, is_train)
    try:
        validated_data = DataValidator(data=data)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Data validation failed: {str(exc)}") from exc
