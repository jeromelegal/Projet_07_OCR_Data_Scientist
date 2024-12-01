import requests
import pytest
import pandas as pd
from api.main import format_data_for_api, predict
import json

##############################################################################################
### variables

API_URL = "https://api-container.germanywestcentral.azurecontainer.io:8501"
MLFLOW_URL = "http://mlflowjlg-container.germanywestcentral.azurecontainer.io:5000/invocations"

@pytest.fixture
def expected_json():
    expected_json_path = "/app/tests/fixtures/expected_data.json"
    with open(expected_json_path, "r") as json_file:
        return json.load(json_file)

##############################################################################################
### test functions 

# verify if Azure Web App is correctly booted and API accessible
def test_app_started():
    try:
        response = requests.get(API_URL)  
        assert response.status_code == 200, "API is not accessible (status not : 200)."
        assert "Streamlit" in response.text, "Streamlit is not correctly booted"
        #assert "Prêt à dépenser" in response.text, "Streamlit is not correctly booted"

    except requests.exceptions.RequestException as e:
        pytest.fail(f"L'application n'est pas accessible : {e}")

# verify if ACR is correctly booted accessible
def test_container_accessibility(expected_json):
    try:

        response = requests.post(MLFLOW_URL, json=expected_json)  
        assert response.status_code == 200, "MLflow container is not accessible."
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Impossible to connect to MLflow container : {e}")

# testing format function in API
def test_format_data_for_api(expected_json):
    # load test_data file
    test_csv_path = "/app/tests/fixtures/test_data.csv"
    df = pd.read_csv(test_csv_path)

    result = format_data_for_api(df)

    # verify json file
    assert result == expected_json, "JSON formatting is not correct."

# testing predict function and verify the 2 prediction results
def test_predict():
    # load test_data file
    test_csv_path = "/app/tests/fixtures/test_data.csv"
    df = pd.read_csv(test_csv_path)

    # JSON convert
    test_data = format_data_for_api(df)

    try:
        # request
        response = requests.post(MLFLOW_URL, json=test_data)
        assert response.status_code == 200, "MLflow container did not send statut HTTP 200."
        predictions = response.json().get("predictions")
        assert predictions is not None, "There is no prediction."

        # Vérifier les prédictions attendues (0 et 1)
        assert len(predictions) == 2, "Quantity of predisction is incorrect."
        assert predictions[0] == 0, "First prediction is Fail, must be : 0."
        assert predictions[1] == 1, "Second prediction is Fail, must be : 1."

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Issue in communication with MLflow container : {e}")