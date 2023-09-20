import os
from unittest.mock import MagicMock, patch

import pytest

from src.serve_utils import (
    create_predictions_response,
    generate_unique_request_id,
    get_model_resources,
)


@pytest.fixture
def resources_paths():
    """Define a fixture for the paths to the test model resources."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_resources_path = os.path.join(cur_dir, "test_resources")
    return {
        "saved_schema_path": os.path.join(test_resources_path, "schema.joblib"),
        "predictor_file_path": os.path.join(test_resources_path, "predictor.joblib"),
        "pipeline_file_path": os.path.join(test_resources_path, "pipeline.joblib"),
        "target_encoder_file_path": os.path.join(
            test_resources_path, "target_encoder.joblib"
        ),
        "model_config_file_path": os.path.join(
            test_resources_path, "model_config.json"
        ),
    }


@pytest.fixture
def model_resources(resources_paths):
    """Define a fixture for the test ModelResources object."""
    return get_model_resources(**resources_paths)


@patch("serve_utils.uuid.uuid4")
def test_generate_unique_request_id(mock_uuid):
    """Test the generate_unique_request_id function."""
    mock_uuid.return_value = MagicMock(hex="1234567890abcdef1234567890abcdef")
    assert generate_unique_request_id() == "1234567890"


@pytest.fixture
def request_id():
    return generate_unique_request_id()


def test_create_predictions_response(predictions_df, schema_provider, request_id):
    """
    Test the `create_predictions_response` function.

    This test checks that the function returns a correctly structured dictionary,
    including the right keys and that the 'status' field is 'success'.
    It also checks that the 'predictions' field is a list, each element of which is a
    dictionary with the right keys.
    Additionally, it validates the 'predictedClass' is among the 'targetClasses', and
    the sum of 'predictedProbabilities' approximates to 1, allowing for a small
    numerical error.

    Args:
        predictions_df (pd.DataFrame): A fixture providing a DataFrame of model
            predictions.
        schema_provider (RegressionSchema): A fixture providing an instance
            of the RegressionSchema.

    Returns:
        None
    """
    response = create_predictions_response(
        predictions_df, schema_provider, request_id, "prediction"
    )

    # Check that the output is a dictionary
    assert isinstance(response, dict)

    # Check that the dictionary has the correct keys
    expected_keys = {
        "status",
        "message",
        "timestamp",
        "requestId",
        "targetDescription",
        "predictions",
    }
    assert set(response.keys()) == expected_keys

    # Check that the 'status' field is 'success'
    assert response["status"] == "success"

    # Check that the 'predictions' field is a list
    assert isinstance(response["predictions"], list)

    # Check that each prediction has the correct keys
    prediction_keys = {"sampleId", "prediction"}
    for prediction in response["predictions"]:
        assert set(prediction.keys()) == prediction_keys
