import pandas as pd
from data.Data_Cleaning import clean_diabetes1
from models.Random_Forest import run_random_forest
from models.KNN_Model import run_knn
from models.KMeans_Model import run_kmeans

def main():
    print("Starting data processing pipeline...")

    # ==========================================
    # Process diabetes1.csv
    # ==========================================
    file1 = 'data/diabetes1.csv'
    print(f"\nProcessing {file1}...")
    
    # Clean Data
    X1, y1 = clean_diabetes1(file_path=file1)
    
    # Save Cleaned Data
    df1_cleaned = X1.copy()
    df1_cleaned['Outcome'] = y1.values
    output_file1 = 'data/diabetes1_Cleaned.csv'
    df1_cleaned.to_csv(output_file1, index=False)
    print(f"-> Saved cleaned data to {output_file1}")

    # Run Random Forest
    run_random_forest(X1, y1, dataset_name="diabetes1_Cleaned")
    
    # Run KNN
    run_knn(X1, y1, dataset_name="diabetes1_Cleaned")
    
    # Run KMeans
    run_kmeans(X1, y1, dataset_name="diabetes1_Cleaned")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
