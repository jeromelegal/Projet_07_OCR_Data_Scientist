# Projet_07_OCR_Data_Scientist

# Credit Scoring Model Deployment

## Overview

This project focuses on building and deploying a **credit scoring system** to estimate whether a client is likely to repay a loan.

It covers the full workflow of a beginner-friendly machine learning product:
- exploratory analysis
- model comparison
- experiment tracking with **MLflow**
- API deployment with **FastAPI**
- a simple **Streamlit** interface
- and local interpretability with **SHAP**

---

## Dataset

The project uses credit application data built from:
- the main application table
- and additional customer-related data sources merged during preprocessing

For deployment, the final API workflow is based on:
- `application_train.csv`
- `application_test.csv`

The target variable is the loan repayment / default outcome.

---

## Method

The workflow of the project was:

1. explore and prepare the credit data
2. build reusable preprocessing functions
3. track experiments with **MLflow**
4. compare several classification models
5. evaluate them with **AUC**, confusion matrix, and a custom **business score**
6. select a final model
7. deploy the scoring pipeline through **MLflow serving + FastAPI + Streamlit**
8. add **SHAP** explanations for local interpretability

---

## Key Results

The project led to four main outcomes:

1. Several models were compared, including **Logistic Regression**, **Random Forest**, **LightGBM**, and **XGBoost**.
2. The evaluation was not limited to standard metrics: a custom **business score** was defined to penalize costly mistakes differently.
3. The final deployed pipeline is based on **LightGBM**, which is also used in the API for SHAP-based explanations.
4. The project was turned into a small end-to-end application with:
   - a **model served through MLflow**,
   - a **FastAPI prediction service**,
   - and a **Streamlit GUI** for CSV upload and prediction display.

Overall, this project shows how a credit scoring model can move from experimentation to a simple deployed product.

---

## Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- imbalanced-learn
- MLflow
- FastAPI
- Streamlit
- SHAP
- Pytest
- Docker
- GitHub Actions
- Jupyter Notebook

---

## Repository Structure

```bash
.
├── .github/workflows/
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── data/source/
├── gui/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── mlflow/
├── modules/
├── notebooks/
│   ├── 0_EDA_MLFlow.ipynb
│   ├── 1_modelisations.ipynb
│   ├── 2_modelisations-cuML.ipynb
│   └── 3_deployment.ipynb
├── tests/
├── Présentation.pdf
├── setup.py
└── README.md
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/jeromelegal/Projet_07_OCR_Data_Scientist.git
cd Projet_07_OCR_Data_Scientist
```

Then install the required dependencies for the API or the interface depending on the component you want to run.

Example:

```bash
pip install -r api/requirements.txt
pip install -r gui/requirements.txt
```

You can then:
- explore the notebooks with Jupyter
- run the API with FastAPI / Uvicorn
- and launch the GUI with Streamlit

---

## Main Takeaways

This project helped me practice:

- classification on imbalanced data,
- business-oriented model evaluation,
- experiment tracking with MLflow,
- model serving,
- API development,
- simple front-end integration for machine learning,
- and prediction interpretability with SHAP.

It was also a strong introduction to the deployment side of a data science project.

---

## Limitations

This project has a few important limitations:
- the interface remains simple,
- deployment is still closer to a learning / portfolio setup than full production,
- and model performance depends heavily on preprocessing and class imbalance handling.

A natural next step would be to improve monitoring, threshold tuning, and production robustness.

---

## Author

**Jérôme Le Gal**  
Data Science student
