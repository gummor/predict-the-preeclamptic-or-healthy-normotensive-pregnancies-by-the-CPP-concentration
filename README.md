# Preeclampsia Risk Prediction API

This repository implements a machine learning pipeline for preeclampsia risk stratification based on age and Copeptin levels.

> **Ethical Statement:**
> All clinical data used in this pipeline consist of synthetic values derived from reported literature means. No Protected Health Information (PHI) or real patient records were utilized. This project is intended for methodology demonstration and educational purposes.

## Project Architecture
- **01_data_generation.py**: Generates a synthetic cohort (n=1000) based on clinical parameters.
- **02_train_model.py**: Executes data preprocessing, feature scaling, and Random Forest classifier training.
- **03_clustering.py**: Performs unsupervised learning (K-Means) for risk group identification.
- **04_app_api.py**: Provides a FastAPI interface for individual and cohort risk assessment.

## Deployment and Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the data-to-model pipeline: `python3 01_data_generation.py && python3 02_train_model.py`
3. Initialize the inference service: `python3 04_app_api.py`

## API Inference Example (CURL)
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
     -H 'Content-Type: application/json' \
     -d '{"Age": 32, "Copeptin": 15.2}'
```
