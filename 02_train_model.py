import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_preeclampsia_model(csv_path='clean_dataset.csv'):
    # Carga de dados
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Erro: clean_dataset.csv não encontrado. Rode o Script 1 primeiro.")
        return

    # Preparação das Features (X) e Alvo (y)
    features = ['Age', 'Iq1', 'Iq3', 'Copeptin']
    X = df[features]
    y = df['Diagnosis_Group']
    weights = df['N']

    # Divisão treino e teste (80/20)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    # Padronização (Essencial para a API)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do Modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Avaliação
    y_pred = model.predict(X_test_scaled)
    print("--- Métricas do Modelo ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Salvando os arquivos para a API (Os artefatos)
    joblib.dump(model, 'preeclampsia_model.joblib')
    joblib.dump(scaler, 'preeclampsia_scaler.joblib')
    print(f"\n✅ Sucesso: Artefatos salvos (.joblib)")

if __name__ == "__main__":
    train_preeclampsia_model()
