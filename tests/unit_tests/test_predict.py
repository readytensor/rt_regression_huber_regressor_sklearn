import numpy as np
import pandas as pd
import pytest

from src.predict import create_predictions_dataframe


def test_create_predictions_dataframe():
    """
    Test the function 'create_predictions_dataframe'.
    Checks if the output is a DataFrame, if its shape and column names are correct,
    and if the ID values match the input.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 1)
    prediction_field_name = "prediction"
    ids = pd.Series(np.random.choice(1000, 5))
    id_field_name = "id"

    df = create_predictions_dataframe(
        predictions_arr,
        prediction_field_name,
        ids,
        id_field_name,
    )

    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame"
    assert df.shape == (5, 2), "Output shape is not correct"
    assert list(df.columns) == [
        id_field_name,
        prediction_field_name,
    ], "Column names are incorrect"
    assert df[id_field_name].equals(ids), "Ids are not correct"


def test_create_predictions_dataframe_mismatch_ids_and_predictions():
    """
    Test the function 'create_predictions_dataframe' for a case where the length of
    the 'ids' series doesn't match the number of rows in 'predictions_arr'.
    Expects a ValueError with a specific message.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 1)
    prediction_field_name = "predicted_class"
    ids = pd.Series(np.random.choice(1000, 4))  # Mismatch in size
    id_field_name = "id"
    with pytest.raises(ValueError) as exception_info:
        _ = create_predictions_dataframe(
            predictions_arr,
            prediction_field_name,
            ids,
            id_field_name,
        )

    assert (
        str(exception_info.value)
        == "Length of ids does not match number of predictions"
    ), "Exception message does not match"
