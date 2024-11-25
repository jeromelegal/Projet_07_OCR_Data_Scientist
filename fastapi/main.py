from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    dataframe_split: dict

app = FastAPI()

# URL de MLflow local
#MLFLOW_URL = "http://192.168.2.189:5000/invocations"

# URL de MLflow Azure
MLFLOW_URL = "http://mlflowjlg.azurecontainer.io:5000/invocations"

@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        # Envoyer les données à MLflow
        response = requests.post(MLFLOW_URL, json=data.dict())
        response.raise_for_status() 

        # Retourner la réponse de MLflow
        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))