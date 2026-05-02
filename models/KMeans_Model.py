from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
import pandas as pd
import numpy as np

def run_kmeans(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a KMeans clustering model on the given features.
    Since KMeans is unsupervised, it clusters without knowing the true labels (y),
    but we map the resulting clusters to the true labels to evaluate accuracy.
    """
    print(f"\n{'='*40}")
    print(f"  KMeans Results: {dataset_name}")
    print(f"{'='*40}")
    
    # 1. Initialize KMeans 
    # n_clusters=2 because we know the data has 2 real classes (Diabetic, Not Diabetic)
    # n_init=10 is used for better stability
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    
    print("Starting KMeans Clustering (Unsupervised)...")
    # 2. Fit the model (Notice we DO NOT pass 'y' here)
    kmeans.fit(X)
    
    # 3. Get cluster labels predicted by KMeans (will be 0 and 1)
    cluster_labels = kmeans.labels_
    
    # 4. Unsupervised Evaluation (Silhouette Score)
    # Measures how well-separated the clusters are mathematically
    sil_score = silhouette_score(X, cluster_labels)
    print(f"\nSilhouette Score: {sil_score:.4f}")
    print("  * (Scores range from -1 to 1. Higher is better, 0 means overlapping clusters)")
    
    # 5. Map clusters to actual true labels for Accuracy calculation
    # KMeans might assign cluster '0' to Diabetic and '1' to Non-Diabetic, or vice-versa.
    # We find the most common true label in each cluster to do a fair comparison.
    mapped_labels = np.zeros_like(cluster_labels)
    for i in range(2):
        mask = (cluster_labels == i)
        if np.sum(mask) > 0: # Ensure the cluster isn't completely empty
            # Find the most frequent true class inside this cluster
            most_frequent_true_label = y[mask].mode()[0]
            mapped_labels[mask] = most_frequent_true_label
            
    # 6. Evaluate Accuracy of the mapped clusters
    accuracy = accuracy_score(y, mapped_labels)
    print(f"\nCluster Accuracy (Mapped): {accuracy * 100:.2f}%\n")
    print("Classification Report (based on mapped clusters):")
    print(classification_report(y, mapped_labels))
    
    print("-" * 40)
    return kmeans
