from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    dataframe_split: dict

app = FastAPI()

# URL de MLflow
MLFLOW_URL = "http://192.168.2.189:5000/invocations"

@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        # Envoyer les données à MLflow
        response = requests.post(MLFLOW_URL, json=data.dict())

        # Vérifier le statut de la réponse
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Erreur depuis MLflow: {response.text}"
            )

        # Retourner la réponse de MLflow
        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))