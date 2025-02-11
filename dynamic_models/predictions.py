import os
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from model_eval import *

# Variables
problem_type = "classification"  # regression vs classification

# Load config file
with open("model_eval_config.json", "r") as config_file:
    CONFIG = json.load(config_file)

# ===============================
# Load Config File Settings
# ===============================
def load_data(problem_type):
    """
    Loads a dataset dynamically from config.json based on problem type.
    """
    if problem_type not in CONFIG:
        raise ValueError(f"Invalid problem_type '{problem_type}' not found in config.")

    dataset_command = CONFIG[problem_type]["default_dataset"]
    target_col = CONFIG[problem_type]["target_col"]

    try:
        # Execute the dataset loading command
        if problem_type == "time-series":
            dataset = sm.datasets.m4_hourly.load_pandas().data  # Specific for time-series
            dataset["date"] = pd.date_range(start="2022-01-01", periods=len(dataset), freq="H")  # Ensure datetime index
            date_column = "date"  # Define the date column for time-series models
        else:
            dataset = eval(dataset_command.split("=")[-1].strip())  # Execute dataset command
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            df[target_col] = dataset.target  # Assign the target column
            date_column = None  # No date column for classification & regression

        return df, date_column, target_col

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset for {problem_type}: {e}")

# ==========================
# Main Execution
# ==========================
# Load the dataset
df, date_column, target_col = load_data(problem_type)

# Print dataset info
print("\nDataframe Info:")
print(df.info())

print("\nData Sample:")
print(df.head())

# Run model evaluation
print("\nRunning Model Evaluation...\n")
evaluator(df, target_column=target_col, problem_type=problem_type, date_column=date_column)

print("\nModel Evaluation Complete! Check the saved CSV files for results.")
