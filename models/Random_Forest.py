from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def run_random_forest(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a simple Random Forest model on the given features and target.
    """
    print(f"\n{'='*40}")
    print(f"  Random Forest Results: {dataset_name}")
    print(f"{'='*40}")
    
    # 1. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Initialize Random Forest Classifier
    # Why n_estimators=100?
    # 100 trees is the default in Scikit-Learn. It is generally a large enough "forest" to 
    # ensure robust, stable predictions (by averaging out the errors of individual trees) 
    # without taking too much computational time.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training Random Forest model...")
    # 3. Train the model
    rf_model.fit(X_train, y_train)
    
    # 4. Make predictions
    y_pred = rf_model.predict(X_test)
    
    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    
    # 6. Feature Importances
    if isinstance(X, pd.DataFrame):
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        print(importances.head(5).to_string(index=False))
        
    # 7. Plot one of the trees from the Random Forest
    plt.figure(figsize=(20, 10))
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    class_names = [str(c) for c in rf_model.classes_]
    
    # Why max_depth=3 in the plot?
    # Similar to Decision Tree, this only limits the visual depth in the saved image
    # to keep the image readable. The actual tree in the forest might be much deeper.
    # We plot the first tree (estimator) in the forest
    plot_tree(rf_model.estimators_[0], feature_names=feature_names, class_names=class_names, filled=True, rounded=True, max_depth=3)
    plt.title(f"Random Forest (Tree 0) for {dataset_name}")
    plt.savefig(f"Random_Forest_{dataset_name}.png", bbox_inches='tight')
    plt.close()
    print(f"-> Saved Random Forest tree plot to Random_Forest_{dataset_name}.png")

    print("-" * 40)
    return rf_model
