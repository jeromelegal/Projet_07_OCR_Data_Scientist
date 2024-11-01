# custom functions to use them in the different notebooks

# import librairies

import mlflow
from mlflow import MlflowClient


# ------------------------------ MLFlow functions ----------------------------
# ----------------------------------------------------------------------------

# mlflow experiment initialization 

def setup_mlflow_experiment():
    """setup an experiment in MLFlow"""

    
    client = MlflowClient(tracking_uri="postgresql://mlflowuser:mlflowuser@localhost/mlflowdb")
    experiment_name = "Scoring_Credit"
    experiment_description = (
    "This is the scoring credit tool project for 'Prêt à dépenser'. "
    )
    experiment_tags = {
        "project_name": "scoring-credit",
        "team": "openclassrooms",
        "project_quarter": "Q4-2024",
        "mlflow.note.content": experiment_description
    }
    experiment = client.get_experiment_by_name(experiment_name)

    # verify if experiment exists
    
    if experiment is not None:
        print(f"experiment_id: {experiment.experiment_id}, \nname: {experiment.name}, \ndescription: {experiment.tags.get('mlflow.note.content', 'Aucune description disponible')}")
        return experiment.experiment_id
    else:
        experiment_id = client.create_experiment(name=experiment_name, tags=experiment_tags)
        print(f"New experiment created : {experiment_id}, \nname : {experiment_name}, \ndescription : {experiment_description}")
        return experiment.experiment_id


# function to logging model

def log_model_with_mlflow(model, model_name, params, metrics, id_experiment):
    with mlflow.start_run(experiment_id=id_experiment):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)


# ------------------------------ others functions ----------------------------
# ----------------------------------------------------------------------------