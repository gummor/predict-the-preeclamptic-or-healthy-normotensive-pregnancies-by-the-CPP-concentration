Copeptin-Based Preeclampsia Risk Prediction Pipeline

AI-driven clinical decision support system integrating unsupervised patient stratification and supervised risk prediction. This repository implements a hybrid machine learning architecture for preeclampsia risk assessment, incorporating insights from our systematic review: "The hypothalamic-neuro-hypophyseal system in preeclampsia: a subgroup meta-analysis of copeptin levels worldwide".

Technical Stack & Hybrid Architecture
● Database Layer: SQLite for relational data warehousing and structured SQL querying.
● Unsupervised Stratification: K-Means (scikit-learn) for unsupervised patient stratification and phenotype subgroup discovery without human labels.
● Supervised Predictive Engine: Random Forest Classifier (scikit-learn) for high-dimensional risk mapping and target classification.
● Serving Backend: FastAPI providing high-performance RESTful JSON inference endpoints.
● Frontend UI & Analytics: Streamlit & Plotly delivering interactive clinical interfaces and live programmatic forest plots.

Pipeline Topology & Module Details
01_data_generation.py → SQLite DB Ingestion (preeclampsia.db)
02_train_model.py → Supervised Classifier Fit (Random Forest)
03_clustering.py → Unsupervised Phenotype Stratification (K-Means)
04_app_api.py → FastAPI Inference Server Backend
05_dashboard.py → Streamlit Analytical Dashboard Front-End
● 01_data_generation.py: Simulates n=1,000 medical cohort records based on literature distributions and persists parameters to SQLite.
● 02_train_model.py: Handles analytical queries, scaling transformations, and exports tree-ensemble configurations for downstream prediction.
● 03_clustering.py: Partitioning protocol to isolate clinical sub-phenotypes and identify unlabelled pathological profiles.
● 04_app_api.py: Exposes headless REST API microservices with native concurrent request validation types.
● 05_dashboard.py: Orchestrates real-time web interface inferences and updates systematic review forest plots dynamically.

Execution & Deployment Guide
# Provision Environment & Initialize Modular Topology Pipeline

pip install -r requirements.txt
python3 01_data_generation.py && python3 02_train_model.py && python3 03_clustering.py
# Launch Backend Serving API & Frontend Interface
uvicorn 04_app_api:app --host 0.0.0.0 --port 8000
streamlit run 05_dashboard.py

Data Governance
All processed records consist of simulated synthetic cohort vectors (n=1,000) generated according to parameters established in peer-reviewed literature. No Protected Health Information (PHI) is processed, ensuring complete compliance with data privacy guidelines.

Software References
● Python Language: Python Software Foundation. (2009). Python 3 Reference Manual.
Available at: https://docs.python.org/3/reference/
● Scikit-learn: Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
● FastAPI: Ramírez, S. (2018). FastAPI framework, high performance, easy to learn, fast to
code, ready for production. GitHub.
● Streamlit: Streamlit Team. (2019). Streamlit: The fastest way to build and share data
apps. GitHub.
● Plotly: Plotly Technologies Inc. (2015). Collaborative web-based graphing and analytics.
Montréal, QC.
● SQLite: Hipp, D. R. (2020). SQLite. Hwaci.
