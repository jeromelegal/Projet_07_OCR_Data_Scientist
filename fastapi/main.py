from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# validate inputs
class PredictionRequest(BaseModel):
    instances: list

# mlflow url
MLFLOW_URL = "http://192.168.2.189:5000/invocations"


@app.post("/predict")
async def predict(data: PredictionRequest):
    response = requests.post(MLFLOW_URL, json=data.dict())
    if response.status_code == 200:
        return {"predictions": response.json()}
    else:
        return {"error": response.text}