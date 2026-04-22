import pandas as pd
import numpy as np
import os

def generate_realistic_dataset(n_samples=1000):
    print("=== Gerando dados sintéticos realistas ===")
    np.random.seed(42)
    
    data = {
        'Age': np.random.normal(28, 6, n_samples).clip(18, 45),
        'Iq1': np.random.normal(0.45, 0.12, n_samples),
        'Iq3': np.random.normal(1.65, 0.35, n_samples),
        'Copeptin': np.random.normal(8.5, 4.2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Lógica de Diagnóstico baseada em thresholds clínicos
    score = (df['Copeptin'] * 0.6) + (df['Iq3'] * 0.3) - (df['Iq1'] * 0.1)
    threshold = np.percentile(score, 80)
    df['Diagnosis'] = (score > threshold).astype(int)
    
    # Adicionando ruído para realismo
    noise = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    df['Diagnosis'] = np.where(noise == 1, 1 - df['Diagnosis'], df['Diagnosis'])
    
    df.to_csv('clean_dataset.csv', index=False)
    print(f"Sucesso: 'clean_dataset.csv' criado com {n_samples} linhas.")

if __name__ == "__main__":
    generate_realistic_dataset()
