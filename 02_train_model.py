import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    df = pd.read_csv('clean_dataset.csv')
    X = df[['Age', 'Copeptin']]
    y = df['Diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, 'modelo_rf_preeclampsia.joblib')
    joblib.dump(scaler, 'scaler_robusto.joblib')
    print("Modelo e Scaler treinados.")

if __name__ == "__main__":
    train_model()
