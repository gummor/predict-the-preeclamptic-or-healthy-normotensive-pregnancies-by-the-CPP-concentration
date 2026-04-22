import os
import joblib

files = [
    'modelo_rf_preeclampsia.joblib',
    'scaler_robusto.joblib',
    '04_app_api.py',
    'requirements.txt'
]

print("--- Verificação de Integridade do Repositório ---")
for f in files:
    if os.path.exists(f):
        print(f"✅ {f} encontrado.")
    else:
        print(f"❌ Erro: {f} NÃO encontrado.")

try:
    model = joblib.load('modelo_rf_preeclampsia.joblib')
    print("✅ Modelo carregado com sucesso.")
except:
    print("❌ Erro ao carregar o modelo. Verifique o nome.")
