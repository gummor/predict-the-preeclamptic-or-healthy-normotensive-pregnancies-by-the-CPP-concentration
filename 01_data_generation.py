import pandas as pd
import numpy as np
import sqlite3

def generate_and_store_data(n_samples=1000):
    np.random.seed(42)
    
    # Metadados Geográficos baseados no seu Forest Plot
    continents = ['Africa', 'Asia', 'Europe', 'North America']
    countries = {'Africa': 'Nigeria', 'Asia': 'China', 'Europe': 'UK', 'North America': 'USA'}
    
    continent_col = np.random.choice(continents, n_samples)
    country_col = [countries[c] for c in continent_col]
    
    age = np.random.normal(28, 6, n_samples).clip(18, 45)
    
    # Simulando níveis mais altos na Ásia e África conforme o seu estudo
    effect_map = {'Africa': 2.73, 'Asia': 3.46, 'Europe': 1.32, 'North America': 0.98}
    effect = np.array([effect_map[c] for c in continent_col])
    
    copeptin_base = np.random.normal(6.0, 2.5, n_samples)
    risk_score = (effect * 1.5) + (age * 0.1) + np.random.normal(0, 2, n_samples)
    diagnosis = (risk_score > np.percentile(risk_score, 75)).astype(int)
    
    copeptin = copeptin_base + (diagnosis * effect * 3)
    copeptin = copeptin.clip(1, 40)
    
    df = pd.DataFrame({
        'Age': age, 
        'Copeptin': copeptin, 
        'Continent': continent_col,
        'Country': country_col,
        'Diagnosis': diagnosis
    })
    
    # Integração SQL: Guardar na Base de Dados SQLite
    conn = sqlite3.connect('preeclampsia.db')
    df.to_sql('patients', conn, if_exists='replace', index=False)
    conn.close()
    
    df.to_csv('clean_dataset.csv', index=False)
    print("Dados geográficos gerados e guardados via SQL ('preeclampsia.db').")

if __name__ == "__main__":
    generate_and_store_data()
