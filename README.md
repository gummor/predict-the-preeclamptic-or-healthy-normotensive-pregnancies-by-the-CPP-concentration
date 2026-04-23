# Preeclampsia Risk Prediction & Meta-Analysis Dashboard

This repository implements a robust machine learning architecture for preeclampsia risk stratification. It integrates real-world findings from our systematic review ("The hypothalamic-neuro-hypophyseal system in preeclampsia: a subgroup meta-analysis of copeptin levels worldwide") into an interactive predictive pipeline.

> **Ethical Statement:**
> All clinical data used for model training consist of values derived from reported literature distributions. No Protected Health Information (PHI) is involved. This project is intended for methodology demonstration and is not a clinical diagnostic tool.

## Technical Stack
- **Database Architecture:** SQLite (Relational storage and SQL querying)
- **Machine Learning Engine:** Random Forest Classifier & K-Means Clustering (scikit-learn)
- **Backend API:** FastAPI (RESTful JSON service)
- **Frontend Dashboard:** Streamlit & Plotly (Interactive visual inferences & Forest Plots)
- **Containerization:** Docker Ready

## Project Structure
- **01_data_generation.py**: Simulates cohort data (n=1000) mapped to continental geographic factors and persists it to a SQLite database (`preeclampsia.db`).
- **02_train_model.py**: Executes SQL queries to extract data, performs feature scaling, and trains the classification model.
- **03_clustering.py**: Performs unsupervised stratification for phenotype identification.
- **04_app_api.py**: Provides a headless FastAPI inference endpoint.
- **05_dashboard.py**: Streamlit application rendering the Forest Plot meta-analysis and providing real-time risk predictions.

## Execution Guide
1. Install dependencies: `pip install -r requirements.txt`
2. Initialize Database & Model: `python3 01_data_generation.py && python3 02_train_model.py`
3. Launch the Interactive Dashboard: `streamlit run 05_dashboard.py`
