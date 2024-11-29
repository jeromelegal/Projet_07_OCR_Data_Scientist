import pytest
import pandas as pd
import mlflow.pyfunc
import numpy as np


# import TEST dataset
@pytest.fixture(scope="module")
def load_data():
    data_path = "../data/source"
    df = pd.read_csv(f"{data_path}/application_test.csv")
    return df 

# import model once
@pytest.fixture(scope="module")
def model():
    artifact_path = "../mlflow/artifacts/LGBM_production_pipeline_model/"
    return mlflow.pyfunc.load_model(artifact_path)

def test_model_prediction(model, load_data):
    prediction = model.predict(load_data)
    assert (np.mean(prediction) < 0.05)
    