import json
import logging
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

config_file = "spark_metadata.json"
etl_pipeline_name = "etl_pipeline_1"
spark_app_name = "MetadataProcessing"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json_config(file_path):
    """
    Loads a JSON configuration file and returns the parsed dictionary.
    
    Args:
        file_path (str): Path to the JSON configuration file.
    
    Returns:
        dict: Parsed configuration data.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def get_azure_secret(vault_name, secret_name):
    """
    Retrieves a secret value from Azure Key Vault.
    
    Args:
        vault_name (str): Name of the Azure Key Vault.
        secret_name (str): Name of the secret to retrieve.
    
    Returns:
        str: Secret value retrieved from Azure Key Vault.
    """
    credential = DefaultAzureCredential()
    keyvault_uri = f"https://{vault_name}.vault.azure.net"
    client = SecretClient(vault_url=keyvault_uri, credential=credential)
    return client.get_secret(secret_name).value


def initialize_spark(settings):
    """
    Initializes and configures a Spark session with specified settings.
    
    Args:
        settings (list): List of Spark configuration settings to apply.
    
    Returns:
        SparkSession: Configured Spark session.
    """
    spark = SparkSession.builder.appName(spark_app_name).getOrCreate()
    for setting in settings:
        exec(setting)
    logging.info("Spark session initialized with settings.")
    return spark


def read_from_source(spark, source_config):
    """
    Reads data from the specified source using Spark and configuration settings.
    
    Args:
        spark (SparkSession): Active Spark session.
        source_config (dict): Configuration details for reading the data source.
    
    Returns:
        DataFrame: Loaded Spark DataFrame.
    """
    read_format = source_config["read_format"]
    options = source_config.get("query_parameters", {})
    df = spark.read.format(read_format).options(**options).load()
    return df


def apply_transformations(df, transformations):
    """
    Applies transformations such as adding columns, sorting, and repartitioning.
    
    Args:
        df (DataFrame): Input Spark DataFrame.
        transformations (dict): Transformation rules to apply.
    
    Returns:
        DataFrame: Transformed Spark DataFrame.
    """
    if "add_columns" in transformations:
        for col_info in transformations["add_columns"]:
            df = df.withColumn(col_info["name"], expr(col_info["expression"]))
    if "sorting" in transformations:
        df = df.orderBy(*[col(c) for c in transformations["sorting"]])
    if "repartition" in transformations:
        df = df.repartition(transformations["repartition"])
    return df


def write_to_adls(df, destination):
    """
    Writes the transformed DataFrame to Azure Data Lake Storage (ADLS) in the specified format.
    
    Args:
        df (DataFrame): DataFrame to be written.
        destination (dict): Destination configuration including file format, compression, partitioning, and path.
    """
    df.write.format(destination["file_format"]).partitionBy(*destination["partitionBy"])
    df.write.mode(destination["mode"]).option("compression", destination["compression"]).save(destination["path"])


def process_table(table_name, table_config, spark):
    """
    Processes a single table by reading from the source, applying transformations, and writing to ADLS.
    
    Args:
        table_name (str): Name of the table being processed.
        table_config (dict): Configuration settings for the table.
        spark (SparkSession): Active Spark session.
    
    Returns:
        str: The name of the successfully processed table.
    """
    df = read_from_source(spark, table_config)
    if "transformations" in table_config:
        df = apply_transformations(df, table_config["transformations"])
    write_to_adls(df, table_config["destination"])
    return table_name


def process_metadata(config_path):
    """
    Main function to orchestrate the ingestion and transformation process based on metadata configuration.
    
    Args:
        config_path (str): Path to the metadata configuration file.
    """
    metadata = load_json_config(config_path)[etl_pipeline_name]
    azure_creds = metadata["azure_creds"]
    key_vault_name = azure_creds["host"]
    sql_host = get_azure_secret(key_vault_name, azure_creds["host"])
    sql_user = get_azure_secret(key_vault_name, azure_creds["user"])
    sql_password = get_azure_secret(key_vault_name, azure_creds["password"])
    spark = initialize_spark(metadata["spark"]["spark_settings"])
    
    graph = nx.DiGraph()
    table_configs = metadata["tables"]
    for table, config in table_configs.items():
        graph.add_node(table)
        for dep in config.get("dependencies", []):
            graph.add_edge(dep, table)
    
    completed = set()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        def can_run(table):
            return all(dep in completed for dep in graph.predecessors(table))
        
        for table in table_configs:
            if can_run(table):
                futures[executor.submit(process_table, table, table_configs[table], spark)] = table
        
        while futures:
            for future in as_completed(futures):
                table = future.result()
                completed.add(table)
                del futures[future]
                for dependent in graph.successors(table):
                    if can_run(dependent):
                        futures[executor.submit(process_table, dependent, table_configs[dependent], spark)] = dependent


process_metadata(config_file)
