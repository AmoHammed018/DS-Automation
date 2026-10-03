# DS Automation

This repository contains a customer churn prediction project using Python and PyCaret. The workflow compares multiple classification models, selects the best-performing one using AUC, tunes it, evaluates the results, and saves the trained pipeline for reuse.

## Files
- `Week_5_assignment_starter.ipynb` — end-to-end notebook for data loading, model comparison, tuning, evaluation, and model export
- `churn_predictor.py` — reusable Python module that loads the saved model and predicts churn probability for new customer data

## Objective
Predict whether a customer is likely to churn based on customer attributes and behavior metrics. The model outputs:
- churn probability
- binary churn prediction
- optional percentile ranking relative to the training distribution

## Workflow
1. Load and inspect churn dataset
2. Prepare and engineer features
3. Compare classification models with PyCaret
4. Select the best model using AUC
5. Tune the model
6. Evaluate performance on a hold-out test set
7. Save the trained pipeline with `joblib`
8. Reuse the trained model in `churn_predictor.py`

## Dependencies
- Python
- pandas
- numpy
- scikit-learn
- PyCaret
- joblib
- matplotlib
- seaborn

## Run
```bash
pip install pandas numpy scikit-learn pycaret joblib matplotlib seaborn
```

Open and run the notebook:

```bash
jupyter notebook Week_5_assignment_starter.ipynb
```

Or run the predictor script:

```bash
python churn_predictor.py
```

The script will prompt for a CSV file path and default to `new_churn_data.csv` if none is provided.

## Detailed Workflow

### `Week_5_assignment_starter.ipynb`

#### Step 1: Import Libraries and Load Data
Imports essential packages (pandas, numpy, matplotlib, seaborn, PyCaret, joblib) and loads the prepared churn dataset from `churn_data_encoded.csv`.

**What happens:**
- Dataset shape, first rows, data types, and null values are inspected
- 7,032 customer records with 13 features are loaded
- The dataset is confirmed to be in the expected format before modeling

#### Step 2: Prepare the Data for Modeling
Removes the `customerID` identifier column and initializes a PyCaret `ClassificationExperiment`.

**What happens:**
- Data is split into 80% training and 20% testing
- Numeric features are normalized
- PyCaret prepares the dataset for classification modeling
- The experiment framework is configured with `Churn` as the target variable

#### Step 3: Compare Multiple Models
Uses PyCaret to evaluate multiple classification algorithms and rank them by AUC.

**What happens:**
- Models tested include: Gradient Boosting, AdaBoost, Logistic Regression, LightGBM, LDA, XGBoost, Random Forest, Naive Bayes, and others
- Each model is evaluated using cross-validation
- A leaderboard is generated, sorted by AUC
- The best-performing model is selected (Gradient Boosting Classifier with AUC ≈ 0.84)

#### Step 4: Tune the Best Model
Performs hyperparameter tuning on the selected model to improve its AUC score.

**What happens:**
- PyCaret searches across different hyperparameter combinations
- The tuning process optimizes parameters like learning rate, max depth, subsample ratio, and number of estimators
- The tuned model achieves an improved cross-validation AUC
- Best hyperparameters are identified and applied

#### Step 5: Evaluate the Tuned Model
Evaluates the final tuned model using multiple metrics on the hold-out test set.

**What happens:**
- Accuracy, precision, recall, F1-score, and ROC-AUC are computed
- A confusion matrix and classification report are generated
- The model achieves approximately 79% accuracy and 83% ROC-AUC
- Performance is validated on unseen data

#### Step 6: Save the Trained Model Pipeline
Saves the complete trained pipeline (preprocessing + classifier) to disk using joblib.

**What happens:**
- The full pipeline is serialized as `churn_model.joblib`
- Preprocessing transformations (scaling, encoding) are preserved
- The trained classifier weights are saved
- The file can be loaded later without retraining

#### Step 7: Create the Reusable Python Module
Generates the `churn_predictor.py` module for production predictions.

**What happens:**
- The `ChurnPredictor` class is defined
- Methods for data preparation, prediction, and file I/O are implemented
- The module is designed to be imported or run as a standalone script

---

### `churn_predictor.py`

#### Class: `ChurnPredictor`
A reusable Python class that encapsulates model loading and prediction logic.

#### `__init__(self, model_path='churn_model.joblib')`
Loads the saved model pipeline from disk.

**What happens:**
- `joblib.load()` deserializes the trained model
- The pipeline is stored in `self.pipeline`
- The object is now ready to make predictions

#### `prepare_data(df)` — Static Method
Transforms raw input data into the exact format expected by the trained model.

**What happens:**
- Converts `TotalCharges` to numeric format
- Removes `customerID` if present
- Creates engineered features if missing:
  - `AvgMonthlyCharge` = TotalCharges / tenure
  - `TenureMonthlyRatio` = tenure / MonthlyCharges
  - `LifetimeValue` = MonthlyCharges × tenure
- Creates categorical bins:
  - `TenureBin`: groups tenure into ranges (0-6, 6-12, 12-24, 24-48, 48-72 months)
  - `ChargeBin`: groups monthly charges into ranges (0-30, 30-50, 50-70, 70-90, 90-120)
- Removes the target column `Churn` if present
- Validates that all required features are present
- Returns data in the correct feature order

#### `predict(self, df, prepare=True)`
Generates churn predictions for a DataFrame.

**What happens:**
- Calls `prepare_data()` if `prepare=True` to ensure input consistency
- Runs `pipeline.predict_proba()` to calculate churn probability for each row
- Creates `Churn_Probability` column (0.0 to 1.0)
- Creates `Churn_Prediction` column (1 if probability ≥ 0.5, else 0)
- Optionally calculates `Churn_Percentile` if training probabilities are available
- Returns the results DataFrame with all original columns plus predictions

#### `predict_file(self, file_path)`
Loads a CSV file and generates predictions.

**What happens:**
- Checks if the file exists
- Prints file metadata (path, row count, column count)
- Reads the CSV into a pandas DataFrame
- Calls `predict()` with `prepare=True`
- Returns the prediction results

#### `print_predictions(self, df, prepare=True)`
Convenience method to print predictions in a readable format.

**What happens:**
- Calls `predict()` to generate predictions
- Prints a table with `Churn_Probability` and `Churn_Prediction` columns
- If `Churn_Percentile` is available, prints it as well
- Returns the prediction results DataFrame

#### `predict_file_interactive()`
Prompts the user for a CSV file path interactively.

**What happens:**
- Asks the user: "Enter the path to the churn CSV file [default: new_churn_data.csv]: "
- Uses the entered path or defaults to `new_churn_data.csv`
- Calls `predict_file()` with the chosen path
- Prints the results using `print_predictions_from_results()`
- Returns the prediction DataFrame

#### `print_predictions_from_results(results)` — Static Method
Prints an existing prediction DataFrame.

**What happens:**
- Displays `Churn_Probability` and `Churn_Prediction` columns
- Displays `Churn_Percentile` if available
- Provides a human-readable summary of predictions

#### CLI Execution Block
When the script is run directly as `python churn_predictor.py`:

**What happens:**
- Instantiates a `ChurnPredictor` object with the saved model
- Prompts the user for a CSV file path (defaults to `new_churn_data.csv`)
- Loads the CSV and generates predictions
- Prints the prediction results to the console

---

## Data Flow Summary

```
Raw Churn Data
    ↓
Data Cleaning & Feature Engineering (Notebook Step 1-2)
    ↓
Model Comparison & Selection (Notebook Step 3)
    ↓
Hyperparameter Tuning (Notebook Step 4)
    ↓
Model Evaluation (Notebook Step 5)
    ↓
Save Pipeline as churn_model.joblib (Notebook Step 6)
    ↓
Load Model in ChurnPredictor (churn_predictor.py)
    ↓
Prepare New Data (churn_predictor.prepare_data)
    ↓
Generate Churn Predictions (churn_predictor.predict)
    ↓
Output Results (Churn_Probability, Churn_Prediction, Churn_Percentile)
```

---

## Notes
- The notebook is the training and experimentation environment.
- The Python script is the deployment/productionized version.
- Model persistence is important because it preserves preprocessing and trained logic.
- This project is a classic supervised binary classification task: predicting whether a customer is likely to churn.
