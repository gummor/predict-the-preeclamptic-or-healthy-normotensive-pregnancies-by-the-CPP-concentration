import pandas as pd
import numpy as np

def generate_realistic_dataset(n_samples=1000):
    print("=== Gerando dados sintéticos: Cohort de Copeptina ===")
    np.random.seed(42)
    age = np.random.normal(28, 6, n_samples).clip(18, 45)
    copeptin = np.random.normal(8.5, 4.2, n_samples).clip(2, 40)
    score = (copeptin * 0.7) + (age * 0.1)
    threshold = np.percentile(score, 80)
    diagnosis = (score > threshold).astype(int)
    df = pd.DataFrame({'Age': age, 'Copeptin': copeptin, 'Diagnosis': diagnosis})
    df.to_csv('clean_dataset.csv', index=False)
    print(f"Sucesso: 'clean_dataset.csv' criado com 1000 amostras.")

if __name__ == "__main__":
    generate_realistic_dataset()
