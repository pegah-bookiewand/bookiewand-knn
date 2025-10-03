import os
import logging
from typing import List, Dict

from sklearn.neighbors import KNeighborsClassifier

from ai.utils.initialization import logger
from ai.utils.data_collection import (
    data_collection_step,
    fetch_tenant_ids,
    load_tenant_table_data
)
from ai.utils.data_registeration import register_data_to_local_storage, register_data_to_s3
from ai.ml.knn.train_preprocessing import preprocessing_step
import shutil


def train_step(
        bucket_name: str,
        run_timestamp: str,
        base_data_dir: str,
        raw_data_prefix: str,
        checkpoints_dir: str,
        logs_dir: str,
        table_columns: Dict[str, List[str]],
        table_categorical_features: Dict[str, List[str]],
        table_numerical_features: Dict[str, List[str]],
        table_target_variables: Dict[str, str],
        table_date_columns: Dict[str, str],
        table_n_neighbors: Dict[str, int],
        table_knn_weight_modes: Dict[str, str],
        table_filter_low_frequency_class_counts: Dict[str, bool],
        table_model_selection_version: Dict[str, str],
        random_seed: int = 42,
) -> None:
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
    skipped_table_tenant_id_tuples = []
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
                                                               RUN_CHECKPOINTS_TENANT_ID_TABLE_PREFIX,)
            os.makedirs(RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR, exist_ok=True)
            logger.info("Checkpoint directory created at: %s", RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR)

            # ======== Table Data Loading ========
            table_data = load_tenant_table_data(
                tenant_id=tenant_id,
                table_name=table_name,
                raw_table_dir=raw_table_dir,
                table_columns=table_columns,
                table_categorical_features=table_categorical_features,
                table_numerical_features=table_numerical_features,
                table_target_variables=table_target_variables,
                table_date_columns=table_date_columns,
                table_filter_low_frequency_class_counts=table_filter_low_frequency_class_counts,
                table_n_neighbors=table_n_neighbors,
                table_knn_weight_modes=table_knn_weight_modes
            )

            # ======== Data Preprocessing ========
            processed_data_dict = preprocessing_step(
                raw_df=table_data['raw_df'],
                target_variable=table_data['target_variable'],
                categorical_features=table_data['categorical_features'],
                numerical_features=table_data['numerical_features'],
                filter_low_frequency_class_counts=table_data['filter_low_frequency_class_counts'],
                random_seed=random_seed,
                logger=logger
            )

            if processed_data_dict["preprocessing_state"] == "EMPTY_DATAFRAME":
                reason = "Processed dataframe is empty, i.e., there are no available records in it."
                logger.warning(f"Skipping training for table {table_name} and tenant {tenant_id}: "
                               f"{reason}")
                skipped_table_tenant_id_tuples.append(
                    [table_name, tenant_id, processed_data_dict["preprocessing_state"], reason]
                )
                continue
            if processed_data_dict["preprocessing_state"] == "EMPTY_FEATURE_MATRIX":
                reason = "Feature matrix is empty after preprocessing."
                logger.warning(f"Skipping training for table {table_name} and tenant {tenant_id}: "
                               f"{reason}")
                skipped_table_tenant_id_tuples.append(
                    [table_name, tenant_id, processed_data_dict["preprocessing_state"], reason])
                continue
            if processed_data_dict["preprocessing_state"] == "CONSTANT_TARGET_VARIABLE":
                reason = "Target variable is constant, i.e., there is only one class label in the target variable."
                logger.warning(f"Skipping training for table {table_name} and tenant {tenant_id}: "
                               f"")
                skipped_table_tenant_id_tuples.append(
                    [table_name, tenant_id, processed_data_dict["preprocessing_state"], reason]
                )
                continue
            min_samples_required = max(3, table_data['n_neighbors'])
            if processed_data_dict["X"].shape[0] < min_samples_required:
                reason = (f"Insufficient samples for KNN training. "
                          f"Required: {min_samples_required}, Available: {processed_data_dict['X'].shape[0]}")
                logger.warning(f"Skipping training for table {table_name} and tenant {tenant_id}: {reason}")
                skipped_table_tenant_id_tuples.append(
                    [table_name, tenant_id, "INSUFFICIENT_SAMPLES", reason]
                )
                continue

            # ======== Model Training on the Entire Dataset  ========
            X = processed_data_dict["X"]
            y = processed_data_dict["y"]
            logger.info("Processed input features: %s", X.shape)

            # Get the filtered DataFrame from preprocessing
            train_df = processed_data_dict["train_df"]

            # Initializing and training the final model
            knn = KNeighborsClassifier(n_neighbors=table_data['n_neighbors'],
                                       weights=table_data['knn_weight_mode'])
            knn.fit(X=X, y=y)

            # Compute and log the training accuracy
            train_accuracy = knn.score(X, y)
            logger.info(f"Training accuracy: {train_accuracy:.4f}")

            # ======== Saving the Artifacts and the Model ========
            hyperparameters = {
                "run_timestamp": run_timestamp,
                "version": run_timestamp,
                "table_name": table_name,
                "tenant_id": tenant_id,
                "model_selection_version": table_model_selection_version[table_name],
                "model_name": "sklearn.neighbors.KNeighborsClassifier",
                "raw_data_path": table_data['csv_path'],
                "knn_checkpoints_dir": RUN_CHECKPOINTS_TENANT_ID_TABLE_DIR,
                "random_seed": random_seed,
                "numerical_features": processed_data_dict["numerical_features"],
                "categorical_features": processed_data_dict["categorical_features"],
                "target_variable": table_data['target_variable'],
                "n_neighbors": table_data['n_neighbors'],
                "weights": table_data['knn_weight_mode'],
                "filter_low_frequency_class_counts": table_data['filter_low_frequency_class_counts'],
                "class_counts": processed_data_dict["class_counts"],
                "removed_classes": processed_data_dict["removed_classes"],
                "constant_columns": processed_data_dict["constant_columns"],
                "num_samples": X.shape[0],
                "input_dim": X.shape[1],
                "X.shape": list(X.shape),
                "y.shape": list(y.shape),
                "train_accuracy": train_accuracy,
            }
            encoders = {
                "numerical_scaler": processed_data_dict["numerical_scaler"],
                "categorical_encoder": processed_data_dict["categorical_encoder"],
                "nan_imputer_value": processed_data_dict["nan_imputer_value"]
            }
            register_data_to_local_storage(raw_df=table_data['raw_df'],
                                           train_df=train_df,
                                           encoders=encoders,
                                           hyperparameters=hyperparameters,
                                           model=knn,
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
    """Clean up local files and directories for a tenant after S3 upload.
    
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