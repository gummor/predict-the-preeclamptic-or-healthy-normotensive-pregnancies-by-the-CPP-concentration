import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_clustering_analysis(csv_path='/Users/gustavo/Documents/triage-predictive-api/clean_dataset.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Erro: clean_dataset.csv não encontrado.")
        return

    # 1. Preparação: Isolamos apenas as variáveis biológicas
    # Ignoramos o Diagnosis_Group (gabarito) para ver se o algoritmo acha os grupos sozinho
    features = ['Age', 'Iq1', 'Iq3', 'Copeptin']
    X = df[features]

    # 2. Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. K-Means (n_clusters=2 para simular Saudável vs Risco)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Validação: Comparando médias de Copeptina por Cluster
    # Geralmente o cluster com Copeptina mais alta é o de risco
    cluster_profile = df.groupby('Cluster')['Copeptin'].mean()
    risk_cluster = cluster_profile.idxmax()

    print("--- Resultados da Clusterização (Não-Supervisionada) ---")
    print(f"Qualidade do Agrupamento (Silhouette Score): {silhouette_score(X_scaled, df['Cluster']):.4f}")
    print("\nMédia de Copeptina por Cluster:")
    print(cluster_profile)
    
    # 5. Comparação com o Diagnosis_Group original
    correct_identification = (df['Cluster'] == df['Diagnosis_Group']).mean()
    if correct_identification < 0.5: # Inverte caso o label do cluster tenha ficado trocado
        correct_identification = 1 - correct_identification
        
    print(f"\nConcordância entre Cluster e Gabarito: {correct_identification*100:.2f}%")
    print(f"✅ Sucesso: Perfis identificados no Cluster {risk_cluster}")

if __name__ == "__main__":
    run_clustering_analysis()
