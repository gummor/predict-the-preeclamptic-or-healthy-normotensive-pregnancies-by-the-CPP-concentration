import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_from_sql():
    # Extração de dados via query SQL
    conn = sqlite3.connect('preeclampsia.db')
    query = "SELECT Age, Copeptin, Diagnosis FROM patients"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    X = df[['Age', 'Copeptin']]
    y = df['Diagnosis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    
    joblib.dump(model, 'modelo_rf_preeclampsia.joblib')
    joblib.dump(scaler, 'scaler_robusto.joblib')
    print("Modelo treinado com sucesso a partir de dados SQL.")

if __name__ == "__main__":
    train_from_sql()
