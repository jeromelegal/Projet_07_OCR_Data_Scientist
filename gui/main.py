# import librairies :
import streamlit as st
import requests
import os

###############################################################################
# variables :

API_URL = "http://api-container.germanywestcentral.azurecontainer.io:8000/predict"


###############################################################################
# fonctions :

def transfer_csv(url, file):
    with open(file, 'rb'):
        files = {'file': (file,'text/csv')}
        response = requests.post(url, files=files)
    if response.status_code != 200:
        print(f"Echec de l'envoi: {response.status_code}, {response.text}")


###############################################################################
# Interface graphique :

st.title("Prêt à dépenser")
st.header("Outil de 'Scoring crédit' :")

uploaded_file = st.file_uploader("Upload csv")
predict_button = st.button("Prédiction")

if uploaded_file is not None:
    csv_file = pd.read_csv(uploaded_file)
    st.write("Dataset chargé.")


if predict_button:
    if uploaded_file is not None:
        response = transfer_csv(API_URL, uploaded_file)
        if response.status_code == 200:
            results = response.json()
            st.write("Résultats de la prédiction :")
            st.json(results)  # Affiche les résultats au format JSON
        else:
            st.error(f"Erreur API ({response.status_code}): {response.text}")
    else:
        st.error('Veuillez charger un fichier CSV avant de lancer la prédiction.')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8501))  
    st.write(f"L'application est déployée sur le port {port}")
    os.system(f"streamlit run main.py --server.port {port} --server.address 0.0.0.0")
