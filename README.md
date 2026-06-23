Copeptin-Based Preeclampsia Risk Prediction Pipeline

AI-driven clinical decision support system integrating unsupervised patient stratification and supervised risk prediction. This repository implements a hybrid machine learning architecture for preeclampsia risk assessment, incorporating insights from our systematic review: "The hypothalamic-neuro-hypophyseal system in preeclampsia: a subgroup meta-analysis of copeptin levels worldwide".

Pipeline Topology & Module Details
01_data_generation.py → SQLite DB Ingestion (preeclampsia.db)

02_train_model.py → Supervised Classifier Fit (Random Forest)

03_clustering.py → Unsupervised Phenotype Stratification (K-Means)

04_app_api.py → FastAPI Inference Server Backend

05_dashboard.py → Streamlit Analytical Dashboard Front-End

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
