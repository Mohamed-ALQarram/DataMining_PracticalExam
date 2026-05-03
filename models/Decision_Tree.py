from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

def run_decision_tree(X, y, dataset_name="Dataset"):
    """
    Trains and evaluates a simple Decision Tree model on the given features and target.
    """
    print(f"\n{'='*40}")
    print(f"  Decision Tree Results: {dataset_name}")
    print(f"{'='*40}")
    
    # 1. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Initialize Base Decision Tree Classifier
    # Why no max_depth specified here? 
    # By default, the tree will grow until all leaves are pure (which may lead to overfitting). 
    # We kept it simple (primitive) here as requested, allowing the tree to fully grow.
    dt_model = DecisionTreeClassifier(random_state=42)
    
    print("Training Decision Tree model...")
    # 3. Train the model
    dt_model.fit(X_train, y_train)
    
    # 4. Make predictions
    y_pred = dt_model.predict(X_test)
    
    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    
    # 6. Feature Importances
    if isinstance(X, pd.DataFrame):
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        print(importances.head(5).to_string(index=False))
        
    # 7. Plot the Decision Tree
    plt.figure(figsize=(20, 10))
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    class_names = [str(c) for c in dt_model.classes_]
    
    # Why max_depth=4 in the plot?
    # This limits the visual depth of the printed image only, not the actual model.
    # We do this so the output image doesn't become gigantically wide and impossible to read.
    plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, max_depth=4)
    plt.title(f"Decision Tree Plot for {dataset_name}")
    plt.savefig(f"Decision_Tree_{dataset_name}.png", bbox_inches='tight')
    plt.close()
    print(f"-> Saved Decision Tree plot to Decision_Tree_{dataset_name}.png")
        
    print("-" * 40)
    return dt_model
