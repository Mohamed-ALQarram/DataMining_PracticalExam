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

### A. K-Nearest Neighbors (KNN) (`KNN_Model.py`)
- **How:** We built a KNN model using a targeted subset of the top features: `['Glucose', 'BMI', 'Age', 'Insulin']`.
- **Why n_neighbors=15:** 15 was chosen to balance between bias and variance. A very small number (like 1 or 3) makes the model highly sensitive to noise (overfitting), while a very large number smooths out predictions too much (underfitting). Furthermore, 15 is an odd number, which is a best practice to avoid voting ties between our 2 classes.

### B. KMeans Clustering (`KMeans_Model.py`)
- **How:** We implemented KMeans and passed the cleaned data to the model *without* the answers (Unsupervised).
- **Why n_clusters=2:** Because we are predicting Diabetes, which is a binary outcome (Diabetic vs Non-Diabetic). We expect the data to naturally group into 2 main clusters.
- **Why max_iter=300:** 300 is the default in Scikit-Learn. It gives the algorithm enough iterations for the cluster centers to stabilize and stop moving without running infinitely.
- **Evaluation:** We mapped the generated clusters to the actual true labels, revealing an impressive **~74% Accuracy**. This proved that our initial data cleaning and scaling steps successfully created a dataset where the classes are naturally separable.
- **Visualization:** We added a scatter plot using `Glucose` and `BMI`. The plot is automatically saved as `kmeans_clusters.png`.

### C. Decision Tree Classifier (`Decision_Tree.py`)
- **How:** We implemented a primitive Decision Tree model on the processed dataset.
- **Why no max_depth during training:** We did not specify a `max_depth` to keep the model simple and primitive, allowing the tree to fully grow until all leaves are pure. 
- **Why max_depth=4 in the plot:** While visualizing the tree using `plot_tree`, we limited the visual depth to 4. This is purely for the output image (`Decision_Tree_diabetes1_Cleaned.png`), so it remains readable and does not become gigantically wide.

### D. Random Forest Classifier (`Random_Forest.py`)
- **How:** We implemented a primitive Random Forest model to identify the most important features. 
- **Why n_estimators=100:** 100 trees is the default in Scikit-Learn. It is generally a large enough "forest" to ensure robust, stable predictions (by averaging out the errors of individual trees) without consuming too much computational time.
- **Why max_depth=3 in the plot:** Similar to the Decision Tree, we limited the visual depth of the first tree (estimator) in the forest to 3 for the saved output image (`Random_Forest_diabetes1_Cleaned.png`) to keep it readable.

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
pip install pandas numpy scikit-learn matplotlib
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
4. Generate and save a scatter plot (`knn_scatter_plot.png`) for the KNN predictions.
5. Generate and save a scatter plot (`kmeans_clusters.png`) visualizing the KMeans clustering results.
