from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carrega o modelo e o scaler atualizados
model = joblib.load(os.path.join(BASE_DIR, 'modelo_rf_preeclampsia.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_robusto.joblib'))

class PredictionInput(BaseModel):
    Age: float
    Copeptin: float

@app.post("/predict")
def predict(data: PredictionInput):
    # Processa apenas os dois parâmetros atuais
    features = np.array([[data.Age, data.Copeptin]])
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    prob = model.predict_proba(features_scaled)[0].tolist()
    
    return {
        "prediction": prediction,
        "probability": prob,
        "recommendation": "Alto Risco" if prediction == 1 else "Baixo Risco"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
