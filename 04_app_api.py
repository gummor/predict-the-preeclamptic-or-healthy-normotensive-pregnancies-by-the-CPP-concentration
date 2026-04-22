from fastapi import FastAPI
  from pydantic import BaseModel
  import joblib
  import numpy as np

  app = FastAPI()

  Carregar o modelo e o scaler usando caminhos absolutos
  model = joblib.load('/Users/gustavo/Documents/triage-predictive-api/preeclampsia_model.joblib')
  scaler = joblib.load('/Users/gustavo/Documents/triage-predictive-api/preeclampsia_scaler.joblib')

  class PredictionInput(BaseModel):
      Age: float
      Iq1: float
      Iq3: float
      Copeptin: float

  @app.post("/predict")
  def predict(data: PredictionInput):
  Preparar dados de entrada
      features = np.array([[data.Age, data.Iq1, data.Iq3, data.Copeptin]])

  Aplicar padronização
      features_scaled = scaler.transform(features)

  Realizar predição
      prediction = int(model.predict(features_scaled)[0])
      probabilities = model.predict_proba(features_scaled)[0].tolist()

      return {
          "prediction": prediction,
          "probability": probabilities
      }

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
