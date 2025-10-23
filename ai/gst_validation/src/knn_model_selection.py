import os
import datetime
import yaml

from dotenv import load_dotenv

from ai.utils.initialization import logger
from ai.ml.knn.model_selection import model_selection_step

load_dotenv(override=True)


def main():
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    # Reading configurations
    GST_VALIDATION_BASE_DATA_DIR = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
    GST_VALIDATION_DATA_DIR = os.getenv('GST_VALIDATION_DATA_DIR')
    GST_VALIDATION_MODEL_SELECTION_DIR = os.getenv('GST_VALIDATION_MODEL_SELECTION_DIR')
    GST_VALIDATION_LOGS_DIR = os.getenv('GST_VALIDATION_LOGS_DIR')
    GST_VALIDATION_BUCKET_NAME = os.getenv('GST_VALIDATION_BUCKET_NAME')
    SPECIFIC_TENANT_ID = os.getenv('SPECIFIC_TENANT_ID', '').strip() or None

    # Check that environment variables are not None
    if not all([GST_VALIDATION_BASE_DATA_DIR,
                GST_VALIDATION_DATA_DIR, GST_VALIDATION_MODEL_SELECTION_DIR, GST_VALIDATION_LOGS_DIR]):
        logger.error("One or more required environment variables are not set.")
        raise ValueError

    # Defining the features and their types
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_selection_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Configuration Loaded:")

    # Reading hyperparameters
    random_seed = config['random_seed']
    logger.info(f"Random Seed: {random_seed}")

    # Accessing table-specific information
    table_columns = {}
    table_categorical_features = {}
    table_numerical_features = {}
    table_target_variables = {}
    table_date_columns = {}
    table_filter_low_frequency_class_counts = {}
    table_must_have_features = {}
    for table in config['tables']:
        table_name = table['table_name']
        columns = table['categorical_columns'] + table['numerical_columns']
        categorical_features = table['categorical_features']
        numerical_features = table['numerical_features']
        target_variable = table['target_variable']
        date_column = table['date_column']
        filter_low_frequency_class_counts = table['filter_low_frequency_class_counts']
        must_have_features = table['must_have_features']

        table_columns[table_name] = columns
        table_categorical_features[table_name] = categorical_features
        table_numerical_features[table_name] = numerical_features
        table_target_variables[table_name] = target_variable
        table_date_columns[table_name] = date_column
        table_filter_low_frequency_class_counts[table_name] = filter_low_frequency_class_counts
        table_must_have_features[table_name] = must_have_features

        logger.info(f"Table Name: {table_name}")
        logger.info(f"- Columns: {columns}")
        logger.info(f"- Categorical Features: {categorical_features}")
        logger.info(f"- Numerical Features: {numerical_features}")
        logger.info(f"- Target Variable: {target_variable}")
        logger.info(f"- Date Column: {date_column}")
        logger.info(f"- Target Variable: {target_variable}")
        logger.info(f"- Filter Low Frequency Class Counts: {filter_low_frequency_class_counts}")
        logger.info(f"- Must Have Features: {must_have_features}")

    # ============ Feature and Model Selection Step ============
    model_selection_step(
        bucket_name=GST_VALIDATION_BUCKET_NAME,
        run_timestamp=run_timestamp,
        base_data_dir=GST_VALIDATION_BASE_DATA_DIR,
        raw_data_prefix=GST_VALIDATION_DATA_DIR,
        model_selection_dir=GST_VALIDATION_MODEL_SELECTION_DIR,
        logs_dir=GST_VALIDATION_LOGS_DIR,
        table_columns=table_columns,
        table_categorical_features=table_categorical_features,
        table_numerical_features=table_numerical_features,
        table_target_variables=table_target_variables,
        table_date_columns=table_date_columns,
        table_filter_low_frequency_class_counts=table_filter_low_frequency_class_counts,
        table_must_have_features=table_must_have_features,
        random_seed=random_seed,
        specific_tenant_id=SPECIFIC_TENANT_ID
    )


if __name__ == "__main__":
    main()
