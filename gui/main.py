# import librairies :
import streamlit as st
import requests
import os
import pandas as pd

###############################################################################
# variables :

API_URL = "http://api-container.germanywestcentral.azurecontainer.io:8000/predict"


###############################################################################
# fonctions :

@st.cache_data
def transfer_csv(url, uploaded_file):
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  # Lève une exception en cas de statut HTTP >= 400
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête vers l'API : {str(e)}")
        return None


###############################################################################
# Interface graphique :

st.title("Prêt à dépenser")
st.header("Outil de 'Scoring crédit' :")

uploaded_file = st.file_uploader("Upload csv")
predict_button = st.button("Prédiction")

if uploaded_file is not None:
    try:
        csv_file = pd.read_csv(uploaded_file)
        st.write("Dataset chargé.")
        st.write(csv_file.head())  # Montre un aperçu des données
    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide ou invalide.")


if predict_button:
    if uploaded_file is not None:
        response = transfer_csv(API_URL, uploaded_file)
        if response and response.status_code == 200:
            results = response.json()
            st.write("Résultats de la prédiction :")
            st.json(results)
        else:
            st.error("Impossible de récupérer une réponse correcte de l'API.")
    else:
        st.error("Veuillez charger un fichier CSV avant de lancer la prédiction.")

