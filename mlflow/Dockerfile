FROM python:3.9-slim

# Installer MLflow et autres dépendances
WORKDIR /app
COPY ./artifacts /app/artifacts  
COPY ./artifacts/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install mlflow[extras]

# Exposer le port pour MLflow
EXPOSE 5000

# Commande pour démarrer le serveur MLflow
CMD ["mlflow", "models", "serve", "-m", "/app/artifacts", "--host", "0.0.0.0", "--port", "5000"]
