# Triage Predictive API - Preeclampsia Risk

Este repositório contém uma solução completa de Machine Learning para triagem de risco de Pré-eclâmpsia, utilizando níveis de Copeptina e Idade como preditores.

> **DISCLAIMER (DATA ETHICS):**
> Os dados clínicos utilizados neste pipeline representam valores sintéticos baseados em médias da literatura científica. Não foram utilizados dados reais de pacientes (PII). Este projeto tem fins estritamente educativos e de demonstração de portfólio.

## Estrutura do Projeto
- **01_data_generation.py**: Gera a cohort sintética (1000 amostras).
- **02_train_model.py**: Treina o modelo Random Forest e salva o Scaler.
- **03_clustering.py**: Realiza análise não supervisionada de grupos de risco.
- **04_app_api.py**: API FastAPI para predição individual ou de cohorts.

## Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Gere os dados e treine o modelo: `python3 01_data_generation.py && python3 02_train_model.py`
3. Inicie a API: `python3 04_app_api.py`

## Exemplo de Teste (Individual)
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
     -H 'Content-Type: application/json' \
     -d '{"Age": 32, "Copeptin": 15.2}'
```
