FROM python:3.9.7-slim

# Installer les outils de compilation et les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libxxhash-dev \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install mlflow and requirements
COPY ./artifacts /app/artifacts
COPY ./artifacts/LGBM_production_pipeline_model/ /app/artifacts/
COPY ./artifacts/LGBM_production_pipeline_model/requirements.txt /app/requirements.txt

# Mettre à jour pip
RUN pip install --upgrade pip

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# (Facultatif) Installer mlflow avec des extras
RUN pip install mlflow[extras]

# Exposer le port pour MLflow
EXPOSE 5000

# Commande pour démarrer le serveur MLflow
CMD ["mlflow", "models", "serve", "--no-conda", "-m", "/app/artifacts", "--host", "0.0.0.0", "--port", "5000"]
