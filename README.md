# Triage Predictive API - Preeclampsia Risk
> **DISCLAIMER (DATA ETHICS):** Dados sintéticos para fins educativos.
## Como Testar
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"Age": 32, "Copeptin": 15.2}'
```
