# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de votre projet dans le conteneur
COPY ./main.py /app/main.py
COPY ./requirements.txt /app/requirements.txt
COPY ./model.pkl /app/model.pkl

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port pour Fastapi
EXPOSE 8000

# Lancer Streamlit
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]