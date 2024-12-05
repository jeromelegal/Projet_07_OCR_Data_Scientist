# Projet_07
OCR Data Scientist projet 7 : **Implémentez un modèle de scoring**

Liens vers l'Interface Graphique : [http://gui-container.germanywestcentral.azurecontainer.io:8501](http://gui-container.germanywestcentral.azurecontainer.io:8501)

## Structure du Projet

Voici une description des principaux dossiers et fichiers de ce projet :

- **.github/workflows/** : Contient les fichiers de configuration pour les workflows GitHub Actions afin d'automatiser des tâches (CI/CD, tests, etc.).
  
- **api/** : Contient les fichiers :
  -  `Dockerfile` : pour la création du container Docker lors du déploiement sur le Cloud.
  -  `main.py` : le code de l'API généré par FastAPI.
  -  `requirements.txt` : liste des librairies qui seront installées dans le container et qui sont nécessaires à l'API.
  
- **data/source/** : Contient le dataset de test qui sert aux tests unitaires.

- **gui/** : contient les fichiers :
  -  `Dockerfile` : pour la création du container Docker lors du déploiement sur le Cloud
  -  `main.py` : le code de l'interface graphique déployée sue le Cloud
  -  `requirements.txt` : liste des librairies qui seront installées dans le container et qui sont nécessaires à l'interface
  
- **mlflow/** : contient le fichier `Dockerfile` afin de créer le container du Modèle, ainsi qu'un dossier `arificats` qui contient les fichiers artifacts de MLFlow Registry.
  
- **modules/** : Librairie Python personnalisée qui contient les fonctions qui reviennent sur plusieurs notebooks.
  
- **notebooks/** : Jupyter Notebooks pour l'exploration, l'analyse des données et les modélisations.
  
- **tests/** : contient les scipts des tests unitaires (model et API/GUI) ainsi que des fichiers "étalon" pour les tests.

- **.gitignore** : liste des fichiers ou dossiers locaux à ne pas prendre en compte par GIT.

- **setup.py** : Script de configuration pour le packaging du projet comme une librairie Python.
  
- **README.md** : (ce fichier) Documentation principale du projet.


## Librairie personnelle :
Dans le dossier "modules", une librairie custom : my_functions.py

Description :
Cette bibliothèque contient des fonctions personnalisées pour le projet Projet_07.