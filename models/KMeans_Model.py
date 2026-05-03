from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
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
    
    # 1. Feature Selection for Clustering
    # Using only the top discriminative features reduces noise from weaker features
    # that confuse distance-based algorithms like KMeans
    top_features = ['Glucose', 'BMI', 'Age', 'Insulin']
    available_features = [f for f in top_features if f in X.columns]
    
    if len(available_features) >= 2:
        X_selected = X[available_features]
        print(f"Features used for KMeans: {available_features}")
    else:
        X_selected = X
        print(f"Using all features: {list(X.columns)}")
    
    # 2. Initialize KMeans 
    # Why n_clusters=2?
    # Because we are predicting Diabetes, which is a binary outcome (Diabetic vs Non-Diabetic). 
    # Therefore, we expect the data to naturally group into 2 main clusters.
    # Why max_iter=300? 
    # 300 is the default in Scikit-Learn. It gives the algorithm enough iterations to converge 
    # (i.e., for the cluster centers to stabilize and stop moving) without running infinitely.
    kmeans = KMeans(n_clusters=2, max_iter=300, random_state=0)
    
    print("Starting KMeans Clustering (Unsupervised)...")
    # 3. Fit the model (Notice we DO NOT pass 'y' here)
    kmeans.fit(X_selected)
    
    # 4. Get cluster labels predicted by KMeans (will be 0 and 1)
    cluster_labels = kmeans.labels_
        
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
    print("-" * 40)
    
    # 7. Visualization
    print("\nGenerating Scatter Plot for KMeans Clusters...")
    # We will use the two most important features (Glucose and BMI) for visualization
    feature1 = 'Glucose'
    feature2 = 'BMI'
    
    # Check if features exist in the selected dataset (fallback to first two columns if not)
    if feature1 not in X_selected.columns or feature2 not in X_selected.columns:
        feature1 = X_selected.columns[0]
        feature2 = X_selected.columns[1]

    plt.figure(figsize=(8, 6))
    
    # Scatter plot for the two clusters
    for i in range(2):
        # Select data points that belong to cluster i
        cluster_data = X_selected[cluster_labels == i]
        
        # Determine the mapped label for this cluster
        if len(mapped_labels[cluster_labels == i]) > 0:
            mapped_label = int(mapped_labels[cluster_labels == i][0])
            label_name = 'Diabetic' if mapped_label == 1 else 'Non-Diabetic'
            label_str = f'Cluster {i} ({label_name})'
        else:
            label_str = f'Cluster {i}'
            
        plt.scatter(cluster_data[feature1], cluster_data[feature2], label=label_str, alpha=0.6, edgecolors='w', s=100)
    
    # Plot cluster centers
    # Note: the centers now correspond to X_selected columns directly
    f1_idx = X_selected.columns.get_loc(feature1)
    f2_idx = X_selected.columns.get_loc(feature2)
    
    centers_f1 = kmeans.cluster_centers_[:, f1_idx]
    centers_f2 = kmeans.cluster_centers_[:, f2_idx]
    
    plt.scatter(centers_f1, centers_f2, c='red', marker='X', s=200, label='Centroids')
    
    plt.title(f'KMeans Clustering Results ({dataset_name})\nFeatures: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'kmeans_clusters.png'
    plt.savefig(plot_filename)
    print(f"-> Saved scatter plot as '{plot_filename}'")
    # plt.show() # Uncomment to display the plot interactively
    
    print("=" * 40)
    return kmeans
