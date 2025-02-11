# ===============================
# Essential Libraries
# ===============================
import os
import numpy as np
import pandas as pd
import random
import warnings
import json

# ===============================
# Data Processing & Utilities
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ===============================
# Evaluation Metrics
# ===============================
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,  # Regression Metrics
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score  # Classification Metrics
)

# ===============================
# Time Series Models
# ===============================
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# ===============================
# Regression Models
# ===============================
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    GradientBoostingRegressor, HistGradientBoostingRegressor, 
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# ===============================
# Classification Models
# ===============================
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

# ===============================
# Boosting Models (Works for Both Regression & Classification)
# ===============================
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


warnings.filterwarnings("ignore")
results_folder = "predictions_performance"

# N_RUNS = 5

# ===============================
# Load Config File Settings
# ===============================
# Load model configuration from config file
with open("model_eval_config.json", "r") as config_file:
    CONFIG = json.load(config_file)

# ===============================
# Dynamic Model Loading
# ===============================
def load_models(problem_type):
    """Loads models dynamically based on the problem type from the config file."""
    if problem_type not in CONFIG:
        raise ValueError(f"Problem type '{problem_type}' not found in config file.")
    
    models_dict = CONFIG[problem_type]["models"]
    models = {name: eval(model_str) for name, model_str in models_dict.items()}
    return models


# Classification Model Performance Metrics
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Metric                     | Definition                                                | Interpretation           |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test Accuracy              | Percentage of correctly classified samples.               | Higher is better         |
# |                            | Measures overall model correctness.                       |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test F1                    | Harmonic mean of Precision & Recall.                      | Higher is better         |
# |                            | Balances false positives & false negatives.               |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test Precision             | True Positives / (True Positives + False Positives).      | Higher is better         |
# |                            | Measures correctness of positive predictions.             |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test Recall                | True Positives / (True Positives + False Negatives).      | Higher is better         |
# |                            | Measures model’s ability to find all actual positives.    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test ROC-AUC               | Area under the ROC curve.                                 | Closer to 1 is better    |
# |                            | Measures how well model separates positive/negative cases.|                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Model                      | Name of the classification model used.                    | Used for comparison      |
# |                            | (e.g., LightGBM, XGBoost, RandomForest).                  |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test Accuracy   | Test Accuracy rescaled between 0 and 1 using MinMax.      | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test F1         | Test F1 rescaled between 0 and 1 using MinMax.            | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test Precision  | Test Precision rescaled between 0 and 1 using MinMax.     | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test Recall     | Test Recall rescaled between 0 and 1 using MinMax.        | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test ROC-AUC    | Test ROC-AUC rescaled between 0 and 1 using MinMax.       | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Overfitting Flag           | Flags models where all normalized metrics are > 0.98.     | True = Overfitting risk  |
# |                            | Indicates that the model may be memorizing training data. |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Golden Score               | Weighted sum of all normalized metrics.                   | Higher is better         |
# |                            | 40% F1 + 25% Accuracy + 15% Precision + 10% Recall +      |                          |
# |                            | 10% ROC-AUC. Penalized by 50% if Overfitting Flag = True. |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
def evaluate_classification_models(df, target_column):
    """Evaluates classification models dynamically based on config."""
    print("Running Classification Model Evaluation...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = load_models("classification")
    
    results = {name: [] for name in models.keys()}

    for run in range(5):  
        random_state = np.random.randint(1, 100)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state, stratify=y)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

            results[name].append({
                "Test Accuracy": test_accuracy,
                "Test F1": test_f1,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test ROC-AUC": test_roc_auc
            })

    results_df = pd.DataFrame([{**metrics, "Model": name} for name, metrics_list in results.items() for metrics in metrics_list])

    # Normalize each metric (higher = better)
    scaler = MinMaxScaler()
    for metric in ["Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"]:
        results_df[f"Normalized {metric}"] = scaler.fit_transform(results_df[[metric]])

    # Overfitting Detection
    results_df["Overfitting Flag"] = results_df.apply(
        lambda row: all(row[f"Normalized {metric}"] > 0.98 for metric in ["Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"]),
        axis=1
    )

    # Compute Golden Score from Config
    gs_weights = CONFIG["classification"]["Golden Score"]
    results_df["Golden Score"] = (
        results_df["Normalized Test F1"] * gs_weights["F1"] +  
        results_df["Normalized Test Accuracy"] * gs_weights["Accuracy"] +  
        results_df["Normalized Test Precision"] * gs_weights["Precision"] +  
        results_df["Normalized Test Recall"] * gs_weights["Recall"] +  
        results_df["Normalized Test ROC-AUC"] * gs_weights["ROC-AUC"]
    )

    # Penalize overfitting models (reduce Golden Score by 50% if flagged)
    results_df.loc[results_df["Overfitting Flag"], "Golden Score"] *= 0.5

    results_df = results_df.sort_values(by="Golden Score", ascending=False)

    os.makedirs(results_folder, exist_ok=True)
    results_df.to_csv(os.path.join(results_folder, "classification_model_performance.csv"), index=False)

    print("\nClassification Model Evaluation Complete! Results saved.")


# Regression Model Performance Metrics
# +-------------------+------------------------------------------------------------+--------------------------+
# | Metric           | Definition                                                 | Interpretation           |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Train MAE        | Mean Absolute Error (MAE) on the training data.            | Lower is better          |
# |                  | Measures average absolute differences between              |                          |
# |                  | predicted and actual values during training.               |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Test MAE         | Mean Absolute Error (MAE) on the test data.                | Lower is better          |
# |                  | Measures how close predictions are to actual values.       |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Test MSE         | Mean Squared Error (MSE) on the test data.                 | Lower is better          |
# |                  | Similar to MAE but squares the errors, penalizing          |                          |
# |                  | larger mistakes more heavily.                              |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Test RMSE        | Root Mean Squared Error (RMSE) on the test data.           | Lower is better          |
# |                  | Square root of MSE, making it interpretable in             |                          |
# |                  | the same units as the target variable.                     |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Test R²          | Coefficient of Determination (R²) on the test data.        | Closer to 1 is better    |
# |                  | Measures how well the model explains variance in           |                          |
# |                  | the actual values.                                         |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Model            | Name of the regression model used.                         | Used for comparison      |
# |                  | (e.g., LightGBM, XGBoost, RandomForest).                   |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Normalized MAE   | MAE rescaled between 0 and 1 using MinMax scaling.         | Lower is better          |
# |                  | Ensures fair comparison across models.                     |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Golden Score     | 1 - Normalized MAE.                                        | Higher is better         |
# |                  | A final ranking metric to compare models.                  |                          |
# |                  | Inverts Normalized MAE so that best models                 |                          |
# |                  | have the highest Golden Score.                             |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
# | Overfitting Flag | Flags models where Train MAE << Test MAE.                  | True = Overfitting risk  |
# |                  | Indicates that the model performs much better on           |                          |
# |                  | training data than test data, suggesting                   |                          |
# |                  | possible overfitting.                                      |                          |
# +-------------------+------------------------------------------------------------+--------------------------+
def evaluate_regression_models(df, target_column):
    """Evaluates regression models dynamically based on config."""
    print("Running Regression Model Evaluation...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = load_models("regression")
    
    results = {name: [] for name in models.keys()}

    for run in range(5):  
        random_state = np.random.randint(1, 100)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)

            results[name].append({
                "Test MAE": test_mae,
                "Test MSE": test_mse,
                "Test R²": test_r2
            })

    results_df = pd.DataFrame([{**metrics, "Model": name} for name, metrics_list in results.items() for metrics in metrics_list])

    # Normalize metrics
    scaler = MinMaxScaler()
    for metric in ["Test MAE", "Test MSE", "Test R²"]:
        results_df[f"Normalized {metric}"] = scaler.fit_transform(results_df[[metric]])

    # Compute Golden Score from Config
    gs_weights = CONFIG["regression"]["Golden Score"]
    results_df["Golden Score"] = (
        (1 - results_df["Normalized Test MAE"]) * gs_weights["MAE"] +
        (1 - results_df["Normalized Test MSE"]) * gs_weights["MSE"] +
        results_df["Normalized Test R²"] * gs_weights["R²"]
    )

    results_df = results_df.sort_values(by="Golden Score", ascending=False)

    os.makedirs(results_folder, exist_ok=True)
    results_df.to_csv(os.path.join(results_folder, "regression_model_performance.csv"), index=False)

    print("\nRegression Model Evaluation Complete! Results saved.")


# Time-Series Model Performance Metrics
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Metric                     | Definition                                                | Interpretation           |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test MAE                   | Mean Absolute Error (MAE) on the test data.               | Lower is better          |
# |                            | Measures the average absolute difference between          |                          |
# |                            | predicted and actual values.                              |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test MSE                   | Mean Squared Error (MSE) on the test data.                | Lower is better          |
# |                            | Similar to MAE but squares the errors, penalizing         |                          |
# |                            | larger mistakes more heavily.                             |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test RMSE                  | Root Mean Squared Error (RMSE) on the test data.          | Lower is better          |
# |                            | Square root of MSE, making it interpretable in            |                          |
# |                            | the same units as the target variable.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Test R²                    | Coefficient of Determination (R²) on the test data.       | Closer to 1 is better    |
# |                            | Measures how well the model explains variance in          |                          |
# |                            | the actual values.                                        |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Model                      | Name of the time-series model used.                       | Used for comparison      |
# |                            | (e.g., ARIMA, SARIMA, Prophet).                           |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test MAE        | Test MAE rescaled between 0 and 1 using MinMax.           | Lower is better          |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test MSE        | Test MSE rescaled between 0 and 1 using MinMax.           | Lower is better          |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test RMSE       | Test RMSE rescaled between 0 and 1 using MinMax.          | Lower is better          |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Normalized Test R²         | Test R² rescaled between 0 and 1 using MinMax.            | Higher is better         |
# |                            | Ensures fair comparison across models.                    |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
# | Golden Score               | Weighted sum of all normalized metrics.                   | Higher is better         |
# |                            | 50% MAE + 30% MSE + 20% R².                               |                          |
# |                            | Penalized if model exhibits extreme errors.               |                          |
# +----------------------------+------------------------------------------------------------+--------------------------+
def evaluate_time_series_models(df, target_column, date_column):
    """Evaluates time-series models dynamically based on config."""
    print("Running Time-Series Model Evaluation...")

    if date_column is None:
        raise ValueError("Time-series models require a 'date_column' parameter.")

    df = df[[date_column, target_column]].dropna().sort_values(by=date_column)
    
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    models = load_models("time-series")
    
    results = {name: [] for name in models.keys()}

    for name, model in models.items():
        try:
            if name == "Prophet":
                train_renamed = train.rename(columns={date_column: "ds", target_column: "y"})
                model.fit(train_renamed)
                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)
                predictions = forecast["yhat"].iloc[-len(test):].values
            else:
                model_fit = model.fit(train[target_column])
                predictions = model_fit.forecast(steps=len(test))

            test_mae = mean_absolute_error(test[target_column], predictions)
            test_mse = mean_squared_error(test[target_column], predictions)
            test_r2 = r2_score(test[target_column], predictions)

            results[name].append({
                "Test MAE": test_mae,
                "Test MSE": test_mse,
                "Test R²": test_r2
            })
        except Exception as e:
            print(f"{name} failed: {e}")

    results_df = pd.DataFrame([{**metrics, "Model": name} for name, metrics_list in results.items() for metrics in metrics_list])

    # Normalize metrics
    scaler = MinMaxScaler()
    for metric in ["Test MAE", "Test MSE", "Test R²"]:
        results_df[f"Normalized {metric}"] = scaler.fit_transform(results_df[[metric]])

    # Compute Golden Score from Config
    gs_weights = CONFIG["time-series"]["Golden Score"]
    results_df["Golden Score"] = (
        (1 - results_df["Normalized Test MAE"]) * gs_weights["MAE"] +
        (1 - results_df["Normalized Test MSE"]) * gs_weights["MSE"] +
        results_df["Normalized Test R²"] * gs_weights["R²"]
    )

    results_df = results_df.sort_values(by="Golden Score", ascending=False)

    os.makedirs(results_folder, exist_ok=True)
    results_df.to_csv(os.path.join(results_folder, "time_series_model_performance.csv"), index=False)

    print("\nTime-Series Model Evaluation Complete! Results saved.")


# ===============================
# Main Evaluator Function
# ===============================
def evaluator(df, target_column, problem_type, date_column=None):
    """Determines and runs the appropriate model evaluation."""
    if problem_type == "classification":
        evaluate_classification_models(df, target_column)
    elif problem_type == "regression":
        evaluate_regression_models(df, target_column)
    elif problem_type == "time-series":
        evaluate_time_series_models(df, target_column, date_column)
    else:
        raise ValueError(f"Invalid problem_type: {problem_type}")
