from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def run_knn(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a KNN model on a specific subset of features requested by the user.
    Uses GridSearchCV to optimize n_neighbors and distance metrics.
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
    
    # 3. Initialize Base KNN Classifier
    knn_base = KNeighborsClassifier()
    
    # 4. Setup Hyperparameter Grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # 5. Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=knn_base, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1,
        scoring='accuracy' # Focusing on overall accuracy for KNN
    )
    
    print("Starting GridSearchCV to find the best hyperparameters for KNN...")
    # 6. Train the model
    grid_search.fit(X_train, y_train)
    
    # 7. Extract the best model
    best_knn_model = grid_search.best_estimator_
    
    print("\nBest Parameters found:")
    print(grid_search.best_params_)
    
    # 8. Make predictions
    y_pred = best_knn_model.predict(X_test)
    
    # 9. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("-" * 40)
    return best_knn_model
