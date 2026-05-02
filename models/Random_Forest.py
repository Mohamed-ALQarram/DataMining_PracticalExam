from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def run_random_forest(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a Random Forest model on the given features and target,
    using GridSearchCV to optimize hyperparameters with a focus on Recall.
    """
    print(f"\n{'='*40}")
    print(f"  Random Forest Results: {dataset_name}")
    print(f"{'='*40}")
    
    # 1. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Initialize Base Random Forest Classifier
    # Added class_weight='balanced' to handle class imbalance (prioritizing the minority class)
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # 3. Setup Hyperparameter Grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # 4. Initialize GridSearchCV
    # We optimize for 'recall' to minimize False Negatives
    grid_search = GridSearchCV(
        estimator=rf_base, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='recall'
    )
    
    print("Starting GridSearchCV to find the best hyperparameters...")
    # 5. Train the model using Grid Search
    grid_search.fit(X_train, y_train)
    
    # 6. Extract the best model and print best parameters
    best_rf_model = grid_search.best_estimator_
    
    print("\nBest Parameters found:")
    print(grid_search.best_params_)
    
    # 7. Make predictions with the best model
    y_pred = best_rf_model.predict(X_test)
    
    # 8. Evaluate the best model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 9. Feature Importances
    # We use the feature importances from the best_estimator_
    if isinstance(X, pd.DataFrame):
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        print(importances.head(5).to_string(index=False))
        
    print("-" * 40)
    return best_rf_model
