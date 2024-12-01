# import librairies :

import pandas as pd
import numpy as np
import streamlit as st
import requests
from pydantic import BaseModel
import os

###############################################################################
# variables :

MLFLOW_URL = "http://mlflowjlg-container.centralus.azurecontainer.io:5000/invocations"
data_json = None

###############################################################################
# fonctions :

class PredictionRequest(BaseModel):
    dataframe_split: dict


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


def predict(data: PredictionRequest):
    try:
        # Envoyer les données à MLflow
        response = requests.post(MLFLOW_URL, json=data)
        response.raise_for_status() 

        # Retourner la réponse de MLflow
        return response.json()

    except requests.exceptions.RequestException as e:
        # Gestion des erreurs liées à la requête HTTP
        st.error(f"Erreur lors de la requête : {e}")
        return None
    except Exception as e:
        # Gestion des autres erreurs
        st.error(f"Une erreur inattendue s'est produite : {e}")
        return None


###############################################################################
# Interface graphique :

st.title("Prêt à dépenser")
st.header("Outil de 'Scoring crédit' :")

uploaded_file = st.file_uploader("Upload csv")
predict_button = st.button("Prédiction")

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    data_json = format_data_for_api(df)
    st.write("Dataset chargé.")


if predict_button:
    if data_json is not None:
        results = predict(data_json)
        st.write(results)
    else:
        st.write('Veuillez charger des données.')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8501))  # Utilise 8501 si PORT n'est pas défini
    st.write(f"L'application est déployée sur le port {port}")
    os.system(f"streamlit run main.py --server.port {port} --server.address 0.0.0.0")
