from typing import Dict
import logging


def load_train_config(
        model_selection_config,
        default_config,
        logger: logging.Logger
):
    """
    Function to load and return the training configuration details for a model.

    Parameters
    ----------
    model_selection_config : dict
        Configuration containing details about model selection, including selected features
        and hyperparameters for specific tables and tenants.

    default_config : dict
        Default configuration containing table information, such as default features,
        hyperparameters, and other metadata like the random seed.

    logger : logging.Logger
        Logger instance for recording function execution details.

    Returns
    -------
    dict
        A dictionary containing training configuration for all specified tables, including:
          - `table_columns`: Mapping of table names to their column lists.
          - `table_categorical_features`: Mapping of table names to their categorical feature lists.
          - `table_numerical_features`: Mapping of table names to their numerical feature lists.
          - `table_target_variables`: Mapping of table names to their target variable names.
          - `table_date_columns`: Mapping of table names to their date column names.
          - `table_n_neighbors`: Mapping of table names to the number of neighbors for KNN models.
          - `table_knn_weight_modes`: Mapping of table names to their weight modes for KNN models.
          - `table_filter_low_frequency_class_counts`: Mapping of table names to their filtering configurations.
          - `random_seed`: Random seed for reproducibility.
    """
    # Accessing table-specific information
    table_columns = {}
    table_categorical_features = {}
    table_numerical_features = {}
    table_target_variables = {}
    table_date_columns = {}
    table_n_neighbors = {}
    table_knn_weights = {}
    table_filter_low_frequency_class_counts = {}
    table_model_selection_version = {}

    for table in default_config['tables']:
        has_model_selection_data = False
        table_name = table['table_name']

        categorical_columns = table['categorical_columns']
        numerical_columns = table['numerical_columns']
        columns = categorical_columns + numerical_columns
        if model_selection_config['model_selection']:
            tenant_id = list(model_selection_config['model_selection'].keys())[0]
            if table_name in model_selection_config['model_selection'][tenant_id]:
                has_model_selection_data = True

        if has_model_selection_data:
            load_minimum_number_of_features = table['load_minimum_number_of_features']
            date_column = table['date_column']
            tenant_id = list(model_selection_config['model_selection'].keys())[0]
            table_model_selection_data = model_selection_config['model_selection'][tenant_id][table_name]

            target_variable = table_model_selection_data['hyperparameters']['target_variable']
            if load_minimum_number_of_features:
                best_config = table_model_selection_data['cv_best_minimal_config']
            else:
                best_config = table_model_selection_data['cv_results'][0]

            features = best_config['features']
            n_neighbors = best_config['n_neighbors']
            weights = best_config['weights']

            categorical_features = [f for f in categorical_columns if f in features and f != target_variable]
            numerical_features = [f for f in numerical_columns if f in features and f != target_variable]

            filter_low_frequency_class_counts = table_model_selection_data['hyperparameters'][
                'filter_low_frequency_class_counts']
            config_version = model_selection_config['model_selection'][tenant_id][table_name][
                'hyperparameters']['version']
        else:
            categorical_features = table['default_categorical_features']
            numerical_features = table['default_numerical_features']
            target_variable = table['target_variable']
            date_column = table['date_column']
            n_neighbors = table['default_n_neighbors']
            weights = table['default_weights']
            filter_low_frequency_class_counts = table['default_filter_low_frequency_class_counts']
            config_version = "default_config"

        table_columns[table_name] = columns
        table_categorical_features[table_name] = categorical_features
        table_numerical_features[table_name] = numerical_features
        table_target_variables[table_name] = target_variable
        table_date_columns[table_name] = date_column
        table_n_neighbors[table_name] = n_neighbors
        table_knn_weights[table_name] = weights
        table_filter_low_frequency_class_counts[table_name] = filter_low_frequency_class_counts
        table_model_selection_version[table_name] = config_version

        logger.info(f"Table Name: {table_name}")
        logger.info(f"- Columns: {columns}")
        logger.info(f"- Categorical Features: {categorical_features}")
        logger.info(f"- Numerical Features: {numerical_features}")
        logger.info(f"- Target Variable: {target_variable}")
        logger.info(f"- Date Column: {date_column}")
        logger.info(f"- N Neighbors: {n_neighbors}")
        logger.info(f"- Weights: {weights}")
        logger.info(f"- Filter Low Frequency Class Counts: {filter_low_frequency_class_counts}")
        logger.info(f"- Model Selection Version: {config_version}")

        # Check if table exists in model selection
        if table_name in model_selection_config["model_selection"]:
            logger.info(f"Using model selection parameters for table {table_name}")
        else:
            logger.info(f"No model selection data found for {table_name}, using defaults")

    train_config = {
        "table_columns": table_columns,
        "table_categorical_features": table_categorical_features,
        "table_numerical_features": table_numerical_features,
        "table_target_variables": table_target_variables,
        "table_date_columns": table_date_columns,
        "table_n_neighbors": table_n_neighbors,
        "table_knn_weight_modes": table_knn_weights,
        "table_filter_low_frequency_class_counts": table_filter_low_frequency_class_counts,
        "table_model_selection_version": table_model_selection_version,
        "random_seed": default_config["random_seed"]
    }
    return train_config


def load_llm_config(
        default_config: Dict,
        logger: logging.Logger
):
    """
    Parse and organize LLM configuration from default configuration dictionary.

    This function extracts table-specific configuration information from a nested configuration
    dictionary and reorganizes it into a structured format suitable for LLM processing workflows.
    It processes multiple tables and their associated metadata including columns, features,
    target variables, date columns, and neighbor parameters.

    Parameters
    ----------
    default_config : Dict
        Dictionary containing configuration data with a 'tables' key that holds a list of
        table configurations. Each table configuration should contain 'table_name',
        'categorical_columns', 'numerical_columns', 'features', 'target_variable',
        'date_column', and 'n_neighbors' keys.
    logger : logging.Logger
        Logger instance for recording configuration loading progress and debugging information.

    Returns
    -------
    Dict
        Dictionary containing organized configuration data with the following keys:
        - 'table_columns': mapping of table names to combined categorical and numerical columns
        - 'table_features': mapping of table names to their feature lists
        - 'table_target_variables': mapping of table names to their target variable names
        - 'table_date_columns': mapping of table names to their date column names
        - 'table_n_neighbors': mapping of table names to their n_neighbors parameter values
    """
    logger.info("Starting LLM configuration loading")

    table_columns = {}
    table_features = {}
    table_constraint_features = {}
    table_target_variables = {}
    table_date_columns = {}
    table_id_columns = {}
    table_n_neighbors = {}
    for table in default_config['tables']:
        table_name = table['table_name']
        categorical_columns = table['categorical_columns']
        numerical_columns = table['numerical_columns']
        columns = categorical_columns + numerical_columns
        features = table['features']
        constraint_features = table['constraint_features']
        target_variable = table['target_variable']
        date_column = table['date_column']
        id_column = table['id_column']
        n_neighbors = table['n_neighbors']

        table_columns[table_name] = columns
        table_features[table_name] = features
        table_constraint_features[table_name] = constraint_features
        table_target_variables[table_name] = target_variable
        table_date_columns[table_name] = date_column
        table_id_columns[table_name] = id_column
        table_n_neighbors[table_name] = n_neighbors

        logger.info(f"Table {table_name} configuration:")
        logger.info(f"  - Categorical columns ({len(categorical_columns)}): {categorical_columns}")
        logger.info(f"  - Numerical columns ({len(numerical_columns)}): {numerical_columns}")
        logger.info(f"  - Total columns: {len(columns)}")
        logger.info(f"  - Features ({len(features)}): {features}")
        logger.info(f"  - Constraint features ({len(table_constraint_features)}): {table_constraint_features}")
        logger.info(f"  - Target variable: {target_variable}")
        logger.info(f"  - Date column: {date_column}")
        logger.info(f"  - ID column: {id_column}")
        logger.info(f"  - N neighbors: {n_neighbors}")

    llm_config = {
        "table_columns": table_columns,
        "table_features": table_features,
        "table_constraint_features": table_constraint_features,
        "table_target_variables": table_target_variables,
        "table_date_columns": table_date_columns,
        "table_id_columns": table_id_columns,
        "table_n_neighbors": table_n_neighbors,
    }
    return llm_config
