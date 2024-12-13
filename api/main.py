###############################################################################
# import libraries

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
import requests
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO
import shap
import pickle
import os


###############################################################################
# instances and classes

app = FastAPI()
data_json = None

class PredictionRequest(BaseModel):
    dataframe_split: dict

###############################################################################
# variables :

MLFLOW_URL = "http://mlflowjlg-container.germanywestcentral.azurecontainer.io:5000/invocations"

###############################################################################
# shap :

model_path = os.path.join(os.getcwd(), "model.pkl")
with open(model_path, "rb") as file:
    pipeline = pickle.load(file)

model_with_threshold = pipeline.named_steps['model_with_threshold']
lgbm_model = model_with_threshold.model  # Accéder au modèle encapsulé
explainer = shap.TreeExplainer(lgbm_model) 

###############################################################################
# functions :

def format_data_for_api(df):
    """
    Format a row in a dataframe to send it to API
    """
    nb_rows = len(df)
    columns = df.columns.tolist()
    
    def cleaning_row(data):
        data = list(data)
        data_cleaned = [
            None if isinstance(x, float) and np.isnan(x)
            else int(x) if isinstance(x, np.integer)
            else float(x) if isinstance(x, np.floating)
            else x
            for x in data
        ]
        return data_cleaned

    return {
        "dataframe_split": {
            "columns": columns,
            "data": [cleaning_row(row) for row in df.itertuples(index=False)]
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier CSV
        file_content = await file.read()
        csv_data = StringIO(file_content.decode('utf-8'))
        df = pd.read_csv(csv_data)

        # Formater les données pour PredictionRequest
        formatted_data = format_data_for_api(df)

        prediction_request = PredictionRequest(**formatted_data)

        # Envoyer les données à MLflow
        response = requests.post(MLFLOW_URL, json=prediction_request.dict())
        response.raise_for_status()
        predictions = response.json()

        # Calculer les valeurs SHAP
        X_transformed = pipeline[:-1].transform(df)
        shap_values = explainer(df)

        # Retourner les résultats de MLflow
        return {"predictions": predictions, "shap_values": shap_values.values.tolist()}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide ou invalide.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la requête vers MLflow : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur inattendue s'est produite : {str(e)}")

    