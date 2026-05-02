import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif

def clean_diabetes1(file_path='diabetes1.csv'):
    """
    Cleans and preprocesses the diabetes1.csv dataset.
    This dataset contains numerical features but has hidden missing values (0s)
    in columns where 0 is biologically impossible.
    """
    # 1. Load the dataset
    df = pd.read_csv(file_path)
    
    # 2. Separate features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 3. Data Cleaning (Handling hidden missing values)
    # Columns where 0 indicates a missing value
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # Replace 0 with NaN so the imputer can recognize them as missing
    X[zero_cols] = X[zero_cols].replace(0, np.nan)
    
    # Initialize SimpleImputer to replace NaN with the median of each column
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 4. Feature Selection
    # Select the top 80% of features based on ANOVA F-value
    selector = SelectPercentile(score_func=f_classif, percentile=80)
    X_selected = selector.fit_transform(X_imputed, y)
    selected_cols = X_imputed.columns[selector.get_support()]
    
    # 5. Data Scaling
    # Use StandardScaler to standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_cols)
    
    return X_scaled_df, y
