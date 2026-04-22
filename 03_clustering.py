import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def perform_clustering():
    print("=== Executando análise de clusters ===")
    df = pd.read_csv('clean_dataset.csv')
    X = df.drop('Diagnosis', axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['Risk_Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_summary = df.groupby('Risk_Cluster').mean()
    print("\nResumo dos Clusters:\n", cluster_summary)
    
    df.to_csv('clean_dataset_clustered.csv', index=False)
    print("Arquivo 'clean_dataset_clustered.csv' gerado.")

if __name__ == "__main__":
    perform_clustering()
