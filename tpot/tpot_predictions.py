import os
import warnings
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing

# Set problem type
problem_type = "classification" 
file_name = f"tpot_{problem_type}_best_pipeline.py"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("tpot").setLevel(logging.ERROR)

# ==========================
# Load Data Function
# ==========================
def load_data(problem_type="classification"):
    """Loads a dataset based on problem type: classification or regression."""
    if problem_type == "classification":
        dataset = load_breast_cancer()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["Target"] = dataset.target  # Binary classification target (0 or 1)
    elif problem_type == "regression":
        dataset = fetch_california_housing()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["Target"] = dataset.target  # Continuous regression target
    else:
        raise ValueError("Invalid problem_type. Choose 'classification' or 'regression'.")
    return df

# ==========================
# Detect Task Type
# ==========================
def detect_problem_type(y):
    """Determines whether the dataset is classification or regression."""
    unique_values = np.unique(y)
    if len(unique_values) > 10:
        return "regression"
    elif len(unique_values) > 2:
        return "multiclass"
    else:
        return "classification"

# ==========================
# Preprocess Data
# ==========================
def preprocess_data(df, target_column):
    """Prepares dataset for modeling."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# ==========================
# TPOT AutoML Model with Dynamic Selection
# ==========================
@ignore_warnings(category=FutureWarning)
@ignore_warnings(category=UserWarning)
def run_tpot_model(X_train, X_test, y_train, y_test, problem_type):
    """Runs TPOT AutoML with optimized settings for faster execution."""
    print(f"\nRunning TPOT AutoML for {problem_type.upper()}... (optimized for speed)")

    # Adjust settings for faster execution
    common_params = {
        "generations": 5,  # Reduce evolution cycles (default: 100)
        "population_size": 20,  # Reduce population per generation (default: 100)
        "mutation_rate": 0.6,  # Lower mutation rate (default: 0.9)
        "crossover_rate": 0.1,  # Lower crossover rate (default: 0.5)
        "n_jobs": -1,  # Use all CPU cores
        "verbosity": 2,  # Show key progress updates
        "max_eval_time_mins": 3,  # Reduce evaluation time per model (default: 5-10)
        "random_state": 42,
    }

    # Select TPOT model based on the problem type
    if problem_type == "regression":
        model = TPOTRegressor(**common_params)
    else:
        model = TPOTClassifier(**common_params)

    model.fit(X_train, y_train)

    print("\nTPOT Model Training Complete (Optimized)!")
    return model

# ==========================
# Extract Model Performance
# ==========================
def extract_model_performance(model):
    """Extracts model comparison details from TPOT's internal logs."""
    evaluated_pipelines = []
    for pipeline_str, pipeline_details in model.evaluated_individuals_.items():
        pipeline_info = {
            "Pipeline": pipeline_str,
            "CV Mean Score": pipeline_details.get("internal_cv_score", np.nan),
            "Preprocessing": pipeline_details.get("pipeline", "N/A")
        }
        evaluated_pipelines.append(pipeline_info)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(evaluated_pipelines)
    
    # Sort by best score
    df_results = df_results.sort_values(by="CV Mean Score", ascending=False)
    
    return df_results

# ==========================
# Save Results
# ==========================
def save_results(df_results):
    """Saves model comparison results to CSV."""
    os.makedirs("tpot_results", exist_ok=True)
    results_path = os.path.join("tpot_results", "model_comparisons.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nModel comparison results saved to: {results_path}")

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # Load Data
    df = load_data(problem_type)

    # Print Dataset Information
    print("\nDataframe Info:")
    print(df.info())

    print("\nData Sample:")
    print(df.head())

    # Detect Problem Type
    detected_type = detect_problem_type(df["Target"])
    print(f"\nDetected Task Type: {detected_type.upper()}")

    # Preprocess Data
    X, y = preprocess_data(df, target_column="Target")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if detected_type != "regression" else None
    )

    # Run TPOT Model
    best_model = run_tpot_model(X_train, X_test, y_train, y_test, detected_type)

    # Extract model performance details
    df_results = extract_model_performance(best_model)
    
    # Save results
    save_results(df_results)

    # Save best model pipeline
    os.makedirs("tpot_models", exist_ok=True)
    best_model.export(os.path.join("tpot_models", file_name))
    print(f"\nBest pipeline saved in `tpot_models/{file_name}`")