# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Copier les fichiers de votre projet dans le conteneur
COPY ./main.py /app/main.py
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port pour Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]