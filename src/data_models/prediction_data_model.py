import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import RegressionSchema


def get_predictions_validator(
    schema: RegressionSchema, prediction_field_name: str
) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame contains the ID field specified in the schema.
    2. That the input DataFrame contains two fields named as target classes.

    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (RegressionSchema): An instance of RegressionSchema.
        prediction_field_name (str): Name of the prediction field.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):

            # Check if DataFrame is empty
            if data.empty:
                raise ValueError(
                    "ValueError: The provided predictions file is empty. "
                    "No scores can be generated. "
                )

            if schema.id not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"ID field '{schema.id}' is not present in predictions file."
                )

            if prediction_field_name not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"Prediction field '{prediction_field_name}' is not present "
                    "in predictions file."
                )
            return data

    return DataValidator


def validate_predictions(
    predictions: pd.DataFrame, data_schema: RegressionSchema, prediction_field_name: str
) -> pd.DataFrame:
    """
    Validates the predictions using the provided schema.

    Args:
        predictions (pd.DataFrame): Predictions data to validate.
        data_schema (RegressionSchema): An instance of
            RegressionSchema.
        prediction_field_name (str): Name of the prediction field

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_predictions_validator(data_schema, prediction_field_name)
    try:
        validated_data = DataValidator(data=predictions)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Prediction data validation failed: {str(exc)}") from exc


if __name__ == "__main__":
    schema_dict = {
        "title": "Regression Smoke Test Dataset",
        "description": "Smoke test dataset for regression task.",
        "modelCategory": "regression",
        "schemaVersion": 1.0,
        "inputDataFormat": {"type": "CSV", "encoding": "utf-8"},
        "id": {"name": "id", "description": "Unique identifier for sample"},
        "target": {
            "name": "target",
            "description": "Target variable",
            "example": 176.7003,
        },
        "features": [
            {
                "name": "number",
                "description": "Synthetic numeric feature",
                "dataType": "NUMERIC",
                "example": 0.2075,
                "nullable": True,
            },
            {
                "name": "color",
                "description": "Synthetic categorical feature",
                "dataType": "CATEGORICAL",
                "categories": ["Blue", "Green", "Red"],
                "nullable": True,
            },
        ],
    }
    schema_provider = RegressionSchema(schema_dict)
    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "number": [2, 4, 6, 8, 9],
            "color": [None, "Blue", "Green", "Red", None],
        }
    )

    validated_data = validate_predictions(predictions, schema_provider)
