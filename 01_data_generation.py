import sqlite3
import pandas as pd
import numpy as np

def generate_clinical_data(num_samples=1000, mode='individual'):
    """
    Gera dados sintéticos baseados em parâmetros clínicos com ruído estatístico.
    Caminho: /Users/gustavo/Documents/triage-predictive-api/01_data_generation.py
    """
    np.random.seed(42)
    
    # Parâmetros Médios (Referência de Literatura Simulada)
    # Formato: [(Média Grupo Saudável, Desvio Padrão), (Média Grupo Risco, Desvio Padrão)]
    params = {
        'Age': [(28, 5), (34, 6)],
        'Iq1': [(1.2, 0.3), (0.8, 0.2)],  
        'Iq3': [(4.2, 0.8), (2.1, 0.5)],  
        'Copeptin': [(4.5, 1.5), (14.2, 4.0)]
    }
    
    data_list = []
    
    for i in range(num_samples):
        # Probabilidade base: 70% saudáveis, 30% em risco
        group = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if mode == 'cohort':
            # Gera baseado apenas na média (baixa variância)
            row = {
                'Age': params['Age'][group][0],
                'Iq1': params['Iq1'][group][0],
                'Iq3': params['Iq3'][group][0],
                'Copeptin': params['Copeptin'][group][0]
            }
        else:
            # Gera valores individuais com distribuição normal (Simula variação entre pacientes)
            row = {
                'Age': np.random.normal(params['Age'][group][0], params['Age'][group][1]),
                'Iq1': np.random.normal(params['Iq1'][group][0], params['Iq1'][group][1]),
                'Iq3': np.random.normal(params['Iq3'][group][0], params['Iq3'][group][1]),
                'Copeptin': np.random.normal(params['Copeptin'][group][0], params['Copeptin'][group][1])
            }
        
        # --- LÓGICA DE RISCO COM INCERTEZA (Realismo Clínico) ---
        # Definimos um score base ponderado
        risk_score = (row['Copeptin'] * 0.5) + (row['Age'] * 0.05)
        
        # Adicionamos um ruído normal para que a fronteira de decisão não seja perfeita
        # Isso gera os "Falsos Positivos" e "Falsos Negativos" necessários para o treino
        noise = np.random.normal(0, 1.8) 
        row['Diagnosis_Group'] = 1 if (risk_score + noise) > 8.0 else 0
        
        # Identificação e peso padrão
        row['Patient_ID'] = i + 1
        row['N'] = 1.0
        data_list.append(row)
        
    df = pd.DataFrame(data_list)
    
    # Garante que não tenhamos valores negativos por causa do ruído (ex: idade negativa)
    df['Age'] = df['Age'].clip(lower=18)
    df['Copeptin'] = df['Copeptin'].clip(lower=0.1)
    
    return df[['Patient_ID', 'Age', 'Iq1', 'Iq3', 'Copeptin', 'Diagnosis_Group', 'N']]

def save_and_export(df):
    # Caminhos absolutos conforme solicitado
    db_path = '/Users/gustavo/Documents/triage-predictive-api/preeclampsia.db'
    csv_path = '/Users/gustavo/Documents/triage-predictive-api/clean_dataset.csv'
    
    conn = sqlite3.connect(db_path)
    df.to_sql('screening_data', conn, if_exists='replace', index=False)
    df.to_csv(csv_path, index=False)
    conn.close()
    
    print(f"✅ Sucesso: Dados realistas gerados.")
    print(f"📊 Distribuição de Risco: {df['Diagnosis_Group'].value_counts(normalize=True)[1]*100:.1f}% dos casos.")

if __name__ == "__main__":
    df_gold = generate_clinical_data(num_samples=1000, mode='individual')
    save_and_export(df_gold)
