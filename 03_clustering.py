import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering():
    df = pd.read_csv('clean_dataset.csv')
    X = df[['Age', 'Copeptin']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Risk_Cluster'] = kmeans.fit_predict(X_scaled)
    df.to_csv('clean_dataset_clustered.csv', index=False)
    print("Clusters gerados.")

if __name__ == "__main__":
    perform_clustering()
