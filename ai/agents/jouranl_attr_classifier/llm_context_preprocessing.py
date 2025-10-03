import os
import logging
import shutil
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ai.utils.initialization import logger
from ai.utils.data_collection import (
    data_collection_step,
    fetch_tenant_ids
)
from ai.ml.knn.preprocessing_utils import preprocess_target_variable
from ai.utils.data_registeration import (
    register_tfidf_data_to_local_storage,
    register_data_to_s3
)


def context_preprocessing_step(
        bucket_name: str,
        run_timestamp: str,
        base_data_dir: str,
        raw_data_prefix: str,
        checkpoints_dir: str,
        logs_dir: str,
        table_features: Dict[str, List[str]],
        table_constraint_features: Dict[str, List[str]],
        table_target_variables: Dict[str, str],
        table_date_columns: Dict[str, str],
        table_id_columns: Dict[str, str],
        table_n_neighbors: Dict[str, int],
        random_seed: int = 42,
) -> None:
    """
    Perform comprehensive data preprocessing for machine learning context including data collection,
    feature validation, TF-IDF vectorization, and model artifact generation for multiple tenants and tables.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket for storing processed data and artifacts
    run_timestamp : str
        Timestamp identifier for the current preprocessing run
    base_data_dir : str
        Base directory path for local data storage
    raw_data_prefix : str
        Prefix for raw data directory structure
    checkpoints_dir : str
        Directory path for storing model checkpoints and artifacts
    logs_dir : str
        Directory path for storing log files
    table_features : Dict[str, List[str]]
        Dictionary mapping table names to their respective feature column lists
    table_constraint_features : Dict[str, List[str]]
        Dictionary mapping table names to their respective constraint feature column lists
        Constraint features are used to filter out records that do not have the exact constraint values
            with query record.
    table_target_variables : Dict[str, str]
        Dictionary mapping table names to their target variable column names
    table_date_columns : Dict[str, str]
        Dictionary mapping table names to their date column names
    table_id_columns : Dict[str, str]
        Dictionary mapping table names to their ID column names
    table_n_neighbors : Dict[str, int]
        Dictionary mapping table names to their k-nearest neighbors parameter values
    random_seed : int, default=42
        Random seed for reproducible results

    Returns
    -------
    None
    """
    absolute_logs_dir = os.path.join(base_data_dir, logs_dir)
    log_file_path = os.path.join(absolute_logs_dir, f"{run_timestamp}.txt")
    os.makedirs(absolute_logs_dir, exist_ok=True)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('[*] %(asctime)s- %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.info("Run timestamp: %s", run_timestamp)

    # ========= data collection step =========
    raw_dataset_dir = os.path.join(base_data_dir, raw_data_prefix, run_timestamp)
    os.makedirs(raw_dataset_dir, exist_ok=True)
    tenant_ids = fetch_tenant_ids()

    artifact_dirs = []
    artifact_prefixes = []
    for tenant_id in tenant_ids:
        data_collection_step(raw_dataset_dir=raw_dataset_dir,
                             tenant_id=tenant_id)

        # extracting tenant tables
        tenant_tables = []
        for table_name in [file for file in os.listdir(raw_dataset_dir) if not file.startswith('.')]:
            table_tenant_ids = [
                t_id.split('.')[0] for t_id in os.listdir(os.path.join(raw_dataset_dir, table_name))
            ]
            if tenant_id in table_tenant_ids:
                tenant_tables.append(table_name)

        # training the classifier for tenant-table
        table_artifact_dirs = []
        table_artifact_prefixes = []
        for table_name in tenant_tables:
            raw_table_dir = os.path.join(raw_dataset_dir, table_name)

            logger.info("========== START PROCESSING TENANT ID: %s ========== for table %s", tenant_id, table_name)
            # training the model for tenant id
            RUN_CHECKPOINTS_TENANT_ID_TABLE_PREFIX = os.path.join(checkpoints_dir,
                                                                  tenant_id,
                                                                  run_timestamp,
                                                                  table_name)
            RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR = os.path.join(base_data_dir,
                                                               RUN_CHECKPOINTS_TENANT_ID_TABLE_PREFIX, )
            os.makedirs(RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR, exist_ok=True)
            logger.info("Checkpoint directory created at: %s", RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR)

            # ======== Table Data Loading ========
            csv_path = os.path.join(raw_table_dir, f'{tenant_id}.csv')
            id_column = table_id_columns[table_name]
            features = table_features[table_name]
            constraint_features = table_constraint_features[table_name]
            target_variable = table_target_variables[table_name]
            date_column = table_date_columns[table_name]
            if target_variable == "account_code":
                target_variable_list = [target_variable, 'account_name']
            else:
                target_variable_list = [target_variable]

            # reading the raw data
            raw_df = pd.read_csv(csv_path)

            # preprocessing the target variables of the raw data
            df = preprocess_target_variable(df=raw_df, target_variable=target_variable, logger=logger)
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values([date_column])
            # Filter features to only include those with sufficient variation
            valid_features = []
            for feat in features:
                unique_count = df[feat].nunique()
                if unique_count > 1:
                    valid_features.append(feat)
                    logger.info(f"Feature '{feat}' is valid with {unique_count} unique values")
                else:
                    logger.warning(f"Excluding feature '{feat}' - only {unique_count} unique value(s)")

            if not valid_features:
                logger.error(f"No valid features found for tenant {tenant_id}, table {table_name}")
                continue  # Skip this tenant-table combination

            df = df[[id_column] + valid_features + constraint_features + [date_column] + target_variable_list]
            df = df.fillna('')
            df.reset_index(drop=True, inplace=True)

            # tokenizing the features
            feature_tokenizer = {}
            feature_token_matrix = {}
            for feat in valid_features:
                feat_tokenizer = TfidfVectorizer(
                    lowercase=True,
                    analyzer='char_wb',
                    ngram_range=(2, 5),
                    min_df=1,
                    max_df=0.95
                )
                feat_tfidf_matrix = feat_tokenizer.fit_transform(df[feat].astype(str))
                feature_tokenizer[feat] = feat_tokenizer
                feature_token_matrix[feat] = feat_tfidf_matrix

            # ======== Saving the Artifacts and the Model ========
            hyperparameters = {
                "run_timestamp": run_timestamp,
                "version": run_timestamp,
                "table_name": table_name,
                "tenant_id": tenant_id,
                "model_name": "sklearn.feature_extraction.text.TfidfVectorizer",
                "raw_data_path": csv_path,
                "knn_checkpoints_dir": RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR,
                "random_seed": random_seed,
                "id_column": id_column,
                "features": valid_features,
                "constraint_features": constraint_features,
                "date_column": date_column,
                "target_variable": target_variable,
                "n_neighbors": table_n_neighbors[table_name],
                "num_samples": len(df),
            }

            register_tfidf_data_to_local_storage(raw_df=raw_df,
                                                 train_df=df,
                                                 feature_tokenizer=feature_tokenizer,
                                                 feature_token_matrix=feature_token_matrix,
                                                 hyperparameters=hyperparameters,
                                                 tenant_checkpoint_dir=RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR,
                                                 logger=logger)

            table_artifact_dirs.append(RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR)
            table_artifact_prefixes.append(RUN_CHECKPOINTS_TENANT_ID_TABLE_PREFIX)
            logger.info(f"Model training for tenant_id={tenant_id} completed successfully.")
        artifact_dirs.append(table_artifact_dirs)
        artifact_prefixes.append(table_artifact_prefixes)

    # registering artifacts to S3
    register_data_to_s3(
        bucket_name=bucket_name,
        run_timestamp=run_timestamp,
        raw_dataset_dir=raw_dataset_dir,
        raw_dataset_prefix=raw_data_prefix,
        artifact_dirs=artifact_dirs,
        artifact_prefixes=artifact_prefixes,
        logger=logger
    )


def _cleanup_tenant_data_from_local_storage(
        tenant_id: str,
        artifact_dirs: List[List[str]],
        table_names: List[str],
        raw_dataset_dir: str,
        logger: logging.Logger
) -> None:
    """
    Clean up local files and directories for a tenant after S3 upload.

    Args:
        tenant_id: The ID of the tenant whose data should be cleaned up
        artifact_dirs: List of directories containing model artifacts
        table_names: List of table names that were processed
        raw_dataset_dir: Directory containing raw data files
        logger: Logger instance for logging cleanup operations
    """
    logger.info(f"Cleaning up local files for tenant {tenant_id}")

    # Clean up model directories
    for dir_path in artifact_dirs:
        for tenant_dir in dir_path:
            if os.path.exists(tenant_dir) and tenant_id in tenant_dir:
                try:
                    shutil.rmtree(tenant_dir)
                    logger.info(f"Removed directory: {tenant_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {tenant_dir}: {str(e)}")

    # Clean up raw data files for this tenant
    for table in table_names:
        raw_tenant_file = os.path.join(raw_dataset_dir, table, f"{tenant_id}.csv")
        if os.path.exists(raw_tenant_file):
            try:
                os.remove(raw_tenant_file)
                logger.info(f"Removed file: {raw_tenant_file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {raw_tenant_file}: {str(e)}")