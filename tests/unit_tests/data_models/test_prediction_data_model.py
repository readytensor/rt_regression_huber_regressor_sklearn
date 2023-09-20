import pandas as pd
import pytest

from src.data_models.prediction_data_model import validate_predictions


def test_validate_data_with_valid_data(schema_provider, predictions_df, model_config):
    """Test the function 'validate_predictions' with valid data."""
    validated_data = validate_predictions(
        predictions_df, schema_provider, model_config["prediction_field_name"]
    )
    assert validated_data is not None
    assert validated_data.shape == predictions_df.shape


def test_validate_data_with_missing_id(schema_provider, predictions_df, model_config):
    predictions_missing_id = predictions_df.drop(columns=["id"])
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(
            predictions_missing_id,
            schema_provider,
            model_config["prediction_field_name"],
        )
    assert (
        "ValueError: Malformed predictions file. "
        "ID field 'id' is not present in predictions file" in str(exc_info.value)
    )


def test_validate_data_with_empty_file(schema_provider, predictions_df, model_config):
    empty_predictions = pd.DataFrame(columns=predictions_df.columns)
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(
            empty_predictions, schema_provider, model_config["prediction_field_name"]
        )
    assert "ValueError: The provided predictions file is empty." in str(exc_info.value)


def test_validate_data_with_prediction_columns(
    schema_provider, predictions_df, model_config
):
    predictions_missing_prediction = predictions_df.drop(columns=["prediction"])
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(
            predictions_missing_prediction,
            schema_provider,
            model_config["prediction_field_name"],
        )
    assert "ValueError: Malformed predictions file. Prediction field " in str(
        exc_info.value
    )
    assert "is not present in predictions file." in str(exc_info.value)
