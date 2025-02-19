## Dyanmically find files in the Git Repo for Databricks
import json
import os
from pyspark.sql import SparkSession

# Ensure dbutils is available
try:
    dbutils = dbutils  # If dbutils exists (inside Databricks), use it
except NameError:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)  # Initialize dbutils if missing

# Get the current notebook path in Databricks
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    print(f"Notebook Path: {notebook_path}")
except Exception as e:
    print(f"Failed to get notebook path: {e}")
    notebook_path = None

# Find the root repo folder by trimming the notebook path
if notebook_path:
    repo_root = "/".join(notebook_path.split("/")[:-1])  # Removes the notebook name
    json_path = f"{repo_root}/configs/config.json"

    # Read the JSON configuration dynamically
    try:
        with open(json_path, "r") as file:
            config_data = json.load(file)
        print("Configuration loaded successfully:", config_data)
    except FileNotFoundError:
        print(f"Configuration file not found: {json_path}")
    except Exception as e:
        print(f"Failed to load JSON configuration: {e}")
else:
    print("Could not determine repo root. Ensure you're running this inside Databricks.")

