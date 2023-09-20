import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing.target_encoder import CustomTargetEncoder


def test_initialization():
    encoder = CustomTargetEncoder(target_field="target")
    assert encoder.target_field == "target"
    assert isinstance(encoder.target_encoder, StandardScaler)


def test_fit():
    encoder = CustomTargetEncoder(target_field="target")
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    encoder.fit(df)
    assert hasattr(encoder, "target_encoder")


def test_transform():
    encoder = CustomTargetEncoder(target_field="target")
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4], "feature": [5, 6, 7, 8, 9]})

    encoder.fit(df)
    transformed = encoder.transform(df)
    assert transformed is not None
    assert transformed.shape == (5, 1)
    assert isinstance(transformed, np.ndarray)


def test_transform_field_not_present():
    encoder = CustomTargetEncoder(target_field="not_present")
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4], "feature": [5, 6, 7, 8, 9]})
    expected_error_msg = (
        "Target field not present in data. "
        ".*Expecting target field: not_present.*Cannot encode target."
    )
    with pytest.raises(Exception, match=expected_error_msg):
        encoder.fit(df)


def test_inverse_transform_1D():
    encoder = CustomTargetEncoder(target_field="target")
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    encoder.fit(df)
    transformed = encoder.transform(df)
    inverse_transformed = encoder.inverse_transform(transformed.flatten())
    np.testing.assert_array_almost_equal(
        inverse_transformed.flatten(), df["target"].values
    )


def test_inverse_transform_2D():
    encoder = CustomTargetEncoder(target_field="target")
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    encoder.fit(df)
    transformed = encoder.transform(df)
    inverse_transformed = encoder.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(
        inverse_transformed.flatten(), df["target"].values
    )


def test_inverse_transform_wrong_shape():
    encoder = CustomTargetEncoder(target_field="target")
    arr = np.array([[0, 1], [2, 3], [4, 5]])

    with pytest.raises(ValueError):
        encoder.inverse_transform(arr)
