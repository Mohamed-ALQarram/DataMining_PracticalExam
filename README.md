# Diabetes Prediction and Analysis Pipeline

## Overview
This project implements a complete machine learning pipeline to predict diabetes using the Pima Indians Diabetes Database (`diabetes1.csv`). The pipeline includes robust data cleaning, feature selection, scaling, and the application of three different machine learning algorithms: Random Forest, K-Nearest Neighbors (KNN), and KMeans Clustering.

## 1. Data Cleaning and Preprocessing (`Data_Cleaning.py`)
### How we cleaned the data:
- **Handling Hidden Missing Values:** In medical datasets like this, a value of `0` for features like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` is biologically impossible. We first replaced these `0`s with `NaN` (Not a Number).
- **Imputation:** We used `SimpleImputer(strategy='median')` to replace the `NaN` values with the median of each respective column.
- **Why we did this:** If we left the `0`s in the dataset, the machine learning algorithms would treat them as actual valid measurements, which would heavily distort the calculations (especially averages and distances). We chose the `median` instead of the `mean` because it is more resistant to extreme outliers in medical data.

### Feature Selection:
- **How:** We used `SelectPercentile(score_func=f_classif, percentile=80)` to select the top 80% most important features based on their statistical relationship with the target variable (`Outcome`).
- **Why:** Removing weak or noisy features helps algorithms learn faster and prevents them from getting confused by irrelevant data.
- **Improvement Note:** Initially, the output of this step dropped the column names (returning a NumPy array). We improved the code to explicitly reconstruct the DataFrame using `selector.get_support()` to preserve feature names, which is crucial for interpreting "Feature Importance" later.

### Data Scaling:
- **How:** We applied `StandardScaler` to the selected features to standardize them (giving them a mean of 0 and variance of 1).
- **Why:** Algorithms like KNN and KMeans rely on measuring the mathematical distance between data points. Since our features have vastly different scales (e.g., Insulin can be in the hundreds, while Diabetes Pedigree Function is a small decimal), scaling ensures that no single feature dominates the distance calculation just because its numbers are larger.

*Note on Project History: We initially started exploring a second dataset (`Diabetes2 .csv`) which included categorical data. For that, we implemented One-Hot Encoding (instead of Label Encoding) to avoid the "Dummy Variable Trap". However, we later decided to drop the second dataset entirely to focus all our enhancements and analysis purely on `diabetes1.csv`.*

## 2. Machine Learning Algorithms

### A. Random Forest Classifier (`Random_Forest.py`)
- **How:** We implemented a Random Forest model using `GridSearchCV` to test 27 different combinations of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`). 
- **Improvement Note:** Initially, the model achieved an accuracy of around 70%. However, we noticed an issue with "Class Imbalance" (there are far more non-diabetic records than diabetic ones). 
- **The Fix:** We improved the model by adding `class_weight='balanced'` and explicitly instructing `GridSearchCV` to optimize for **Recall** (`scoring='recall'`). 
- **Why:** In medical diagnoses, minimizing "False Negatives" (telling a sick patient they are healthy) is the highest priority. By focusing on Recall, the model became highly sensitive to detecting the minority class, jumping to a Recall score of 75% and an overall accuracy of 73.38%. The preserved feature names also allowed us to extract that `Glucose`, `BMI`, and `Age` were the most critical factors.

### B. K-Nearest Neighbors (KNN) (`KNN_Model.py`)
- **How:** We built a KNN model using a targeted subset of the top features: `['Glucose', 'BMI', 'Age', 'Insulin']`. We used `GridSearchCV` to find the best number of neighbors and the best distance metric (it chose `metric='manhattan'` and `n_neighbors=15`).
- **Why:** By restricting the KNN model to only the most powerful, noise-free features, the algorithm could measure distances much more accurately. This specific enhancement allowed the KNN model to achieve a very impressive **79.22% Accuracy**.

### C. KMeans Clustering (`KMeans_Model.py`)
- **How:** We implemented KMeans with `k=2` (representing the two classes: Diabetic and Non-Diabetic). We passed the cleaned data to the model *without* the answers (Unsupervised).
- **Why:** The goal was to see if the data naturally grouped itself into two distinct clusters based purely on the mathematical similarities of the features. 
- **Evaluation:** We used the `Silhouette Score` to measure cluster separation. To understand its real-world performance, we then mapped the generated clusters to the actual true labels, revealing an impressive **71.35% Accuracy**. This proved that our initial data cleaning and scaling steps successfully created a dataset where the classes are naturally separable.

## 3. The Main Pipeline (`main.py`)
- **How:** This is the entry point of the application. It imports the cleaning function and the three algorithm functions. It executes them sequentially and saves the cleaned dataset as `diabetes1_Cleaned.csv`.
- **Why:** This creates a clean, automated, and easy-to-read pipeline that handles data ingestion, processing, modeling, and evaluation in one single run.

## 4. How to Clone and Run Locally

Follow these steps to set up the project and run the pipeline on your local machine:

### Prerequisites
- **Python**: Ensure you have Python 3.8 or higher installed.
- **Git**: Ensure you have Git installed to clone the repository.

### Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Mohamed-ALQarram/DataMining_PracticalExam.git
cd DataMining_PracticalExam
```

### Step 2: Create a Virtual Environment (Recommended)
It is best practice to create a virtual environment to manage dependencies:
```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Required Libraries
The project relies on standard data science libraries. Install them using pip:
```bash
pip install pandas numpy scikit-learn
```

### Step 4: Run the Pipeline
Once the libraries are installed, you can execute the entire pipeline with a single command. Make sure you are in the root directory of the project:
```bash
python main.py
```
This will automatically:
1. Load and clean the data from the `data/` folder.
2. Train and evaluate the Random Forest, KNN, and KMeans models from the `models/` folder.
3. Print all accuracy metrics, classification reports, and feature importances directly to your terminal.
