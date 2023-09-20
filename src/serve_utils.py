"""
This script contains utility functions/classes that are used in serve.py
"""
import uuid
from typing import Any, Dict, Tuple

import pandas as pd

from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from predict import create_predictions_dataframe, predict_and_rescale
from prediction.predictor_model import load_predictor_model
from preprocessing.preprocess import load_pipeline_and_target_encoder, transform_data
from schema.data_schema import load_saved_schema
from utils import read_json_as_dict

logger = get_logger(task_name="serve")


class ModelResources:
    def __init__(
        self,
        saved_schema_dir_path: str,
        model_config_file_path: str,
        preprocessing_dir_path: str,
        predictor_dir_path: str,
    ):
        self.data_schema = load_saved_schema(saved_schema_dir_path)
        self.model_config = read_json_as_dict(model_config_file_path)
        self.predictor_model = load_predictor_model(predictor_dir_path)
        self.preprocessor, self.target_encoder = load_pipeline_and_target_encoder(
            preprocessing_dir_path
        )


def get_model_resources(
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    **kwargs,
) -> ModelResources:
    """
    Returns an instance of ModelResources.

    Args:
        saved_schema_dir_path (str): Dir path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        preprocessing_dir_path (str): he dir path where to save the pipeline
            and target encoder.
        predictor_dir_path (str): Path to the saved predictor model file.
    Returns:
        Loaded ModelResources object
    """
    try:
        model_resources = ModelResources(
            saved_schema_dir_path,
            model_config_file_path,
            preprocessing_dir_path,
            predictor_dir_path,
        )
    except Exception as exc:
        err_msg = "Error occurred loading model for serving."
        # Log the error to the general logging file 'serve.log'
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file 'serve-error.log'
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        raise exc
    return model_resources


def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


async def transform_req_data_and_make_predictions(
    data: pd.DataFrame, model_resources: ModelResources, request_id: str
) -> Tuple[pd.DataFrame, dict]:
    """Transform request data and generate predictions based on request.

    Function performs the following steps:
    1. Validate the request data
    2. Transforms the dataframe using preprocessing pipeline
    3. Makes predictions as np array on the transformed data using the predictor model
    4. Converts predictions np array into pandas dataframe with required structure
    5. Converts the predictions dataframe into a dictionary with required structure

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model_resources (ModelResources): Resources needed by inference service.
        request_id (str): Unique request id for logging and tracking

    Returns:
        Tuple[pd.DataFrame, dict]: Tuple containing transformed data and
            prediction response.
    """
    logger.info(f"Predictions requested for {len(data)} samples...")

    # validate the data
    logger.info("Validating data...")
    validated_data = validate_data(
        data=data, data_schema=model_resources.data_schema, is_train=False
    )

    logger.info("Transforming data sample(s)...")
    transformed_data, _ = transform_data(
        model_resources.preprocessor, model_resources.target_encoder, validated_data
    )

    logger.info("Making predictions...")
    rescaled_predictions_arr = predict_and_rescale(
        model_resources.predictor_model,
        model_resources.target_encoder,
        transformed_data,
    )
    logger.info("Converting predictions array into dataframe...")
    predictions_df = create_predictions_dataframe(
        rescaled_predictions_arr,
        model_resources.model_config["prediction_field_name"],
        data[model_resources.data_schema.id],
        model_resources.data_schema.id,
    )

    logger.info("Converting predictions dataframe into response dictionary...")
    predictions_response = create_predictions_response(
        predictions_df,
        model_resources.data_schema,
        request_id,
        model_resources.model_config["prediction_field_name"],
    )
    return transformed_data, predictions_response


def create_predictions_response(
    predictions_df: pd.DataFrame,
    data_schema: Any,
    request_id: str,
    prediction_field_name: str,
) -> Dict[str, Any]:
    """
    Convert the predictions DataFrame to a response dictionary in required format.

    Args:
        transformed_data (pd.DataFrame): The transfomed input data for prediction.
        data_schema (Any): An instance of the RegressionSchema.
        request_id (str): Unique request id for logging and tracking
        prediction_field_name (str): Name of the prediction field

    Returns:
        dict: The response data in a dictionary.
    """
    sample_predictions = []
    for sample in predictions_df.to_dict(orient="records"):
        sample_predictions.append(
            {
                "sampleId": sample[data_schema.id],
                "prediction": round(sample[prediction_field_name], 5),
            }
        )
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": pd.Timestamp.now().isoformat(),
        "requestId": request_id,
        "targetDescription": data_schema.target_description,
        "predictions": sample_predictions,
    }
    return predictions_response
