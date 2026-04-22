from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Isso faz o Python descobrir onde ele mesmo está guardado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_rf_preeclampsia.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_robusto.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PredictionInput(BaseModel):
    Age: float
    Iq1: float
    Iq3: float
    Copeptin: float

@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[data.Age, data.Iq1, data.Iq3, data.Copeptin]])
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0].tolist()

    return {
        "prediction": prediction,
        "probability": probabilities,
        "recommendation": "Alto Risco" if prediction == 1 else "Baixo Risco"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
