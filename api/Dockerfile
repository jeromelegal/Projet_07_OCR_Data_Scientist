# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de votre projet dans le conteneur
COPY ./main.py /app/main.py
COPY ./requirements.txt /app/requirements.txt
COPY ./model.pkl /app/model.pkl
COPY ./modules/ /app/modules/

# Installer les dépendances
RUN apt-get update && apt-get install -y libgomp1
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port pour Fastapi
EXPOSE 8000

# Lancer Streamlit
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]