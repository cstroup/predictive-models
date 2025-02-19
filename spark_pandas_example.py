import pyspark.pandas as ps

# Read a JSON file using Pandas-like API
df = ps.read_json("abfss://container@storageaccount.dfs.core.windows.net/datalake/config.json")

# Perform transformations (Pandas-like)
df["year"] = df["date"].dt.year
df_grouped = df.groupby("category").count()

df.head()
