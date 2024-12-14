###############################################################################
# import libraries

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
import requests
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO
import shap
import pickle
import modules.my_functions as mf
import json


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

with open("model.pkl", "rb") as file:
    pipeline  = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

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
        shap_values = explainer(X_transformed)
        explained_value = explainer.expected_value

        shap_values_explanation = shap.Explanation(
            values=shap_values.values,
            base_values=explained_value,
            data=X_transformed,
            feature_names=feature_names  # Ajouter les noms des colonnes
        )

        # Retourner les résultats de MLflow
        final_response = {
            "predictions": predictions,  # Ce que retourne MLflow
            "explained_value": explained_value,  # Valeur expliquée par SHAP
            "shap_values": shap_values.values.tolist(),  # Valeurs SHAP sous forme de liste
            "feature_names": shap_values_explanation.feature_names
        }
        return final_response

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide ou invalide.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la requête vers MLflow : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur inattendue s'est produite : {str(e)}")

    