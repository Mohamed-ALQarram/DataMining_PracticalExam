from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import pandas as pd

def run_knn(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a KNN model on a specific subset of features requested by the user.
    Uses a fixed n_neighbors=15 directly.
    """
    print(f"\n{'='*40}")
    print(f"  KNN Results: {dataset_name}")
    print(f"{'='*40}")
    
    # 1. Select specific features as requested
    selected_features = ['Glucose', 'BMI', 'Age', 'Insulin']
    
    # Ensure the columns exist in the DataFrame (defensive programming)
    missing_cols = [col for col in selected_features if col not in X.columns]
    if missing_cols:
        print(f"Error: The following required features are missing: {missing_cols}")
        return None
        
    X_subset = X[selected_features]
    print(f"Features used for KNN: {selected_features}\n")
    
    # 2. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
    
    # 3. Initialize KNN Classifier
    # Why n_neighbors=15?
    # 15 is chosen to balance between bias and variance. A very small number (e.g., 1 or 3) makes the model 
    # highly sensitive to noise (overfitting), while a very large number smooths out predictions too much (underfitting).
    # Also, it's generally good practice to pick an odd number to avoid ties in voting between 2 classes.
    best_knn_model = KNeighborsClassifier(n_neighbors=15)
    
    print("Training KNN Model with n_neighbors=15...")
    # 4. Train the model
    best_knn_model.fit(X_train, y_train)
    
    # 8. Make predictions
    y_pred = best_knn_model.predict(X_test)
    
    # 9. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    print("-" * 40)
    
    # 10. Visualization (Scatter Plot)
    print("\nGenerating Scatter Plot for KNN Predictions...")
    feature1 = 'Glucose'
    feature2 = 'BMI'
    
    # Check if features exist in the selected dataset
    if feature1 not in X_test.columns or feature2 not in X_test.columns:
        feature1 = X_test.columns[0]
        feature2 = X_test.columns[1]

    plt.figure(figsize=(8, 6))
    
    # Scatter plot for the two classes (0 and 1) based on PREDICTIONS
    for i in range(2):
        # Select data points that belong to predicted class i
        class_data = X_test[y_pred == i]
        
        label_name = 'Diabetic' if i == 1 else 'Non-Diabetic'
        label_str = f'Predicted: {label_name}'
            
        plt.scatter(class_data[feature1], class_data[feature2], label=label_str, alpha=0.6, edgecolors='w', s=100)
    
    plt.title(f'KNN Predictions on Test Set ({dataset_name})\nFeatures: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'knn_scatter_plot.png'
    plt.savefig(plot_filename)
    print(f"-> Saved scatter plot as '{plot_filename}'")
    plt.close() # Close figure to free memory
    
    print("=" * 40)
    return best_knn_model
