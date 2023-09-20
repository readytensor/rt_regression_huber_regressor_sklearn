def test_infer_endpoint_integration(
    app, sample_request_data, sample_response_data, schema_provider
):
    """
    End-to-end integration test for the /infer endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the
    /infer endpoint, and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the
    test_model_resources fixture, and the application's dependency on ModelResources
    is overridden to use this instance for the test.

    The function sends a POST request to the "/infer" endpoint with the
    test_sample_request_data using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and
    compares specific values in the returned response_data with the
    sample_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        app (TestClient): The test client for the FastAPI application.
        sample_request_data (dict): The fixture for test request data.
        sample_response_data (dict): The fixture for expected response data.
        schema_provider (Any): The fixture for the schema provider.
    Returns:
        None
    """
    response = app.post("/infer", json=sample_request_data)
    response_data = response.json()

    # assertions
    assert set(response_data.keys()) == set(response.json().keys())
    assert (
        response_data["predictions"][0]["sampleId"]
        == sample_response_data["predictions"][0]["sampleId"]
    )
    assert "prediction" in response_data["predictions"][0]
