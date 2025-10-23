import os
import logging
from typing import List, Dict

from ai.utils.initialization import logger
from ai.utils.data_collection import (
    data_collection_step,
    fetch_tenant_ids,
    load_tenant_table_data
)
from ai.utils.data_registeration import (
    register_model_selection_to_local_storage,
    register_data_to_s3
)
from ai.ml.knn.preprocessing_utils import (
    general_preprocessing
)
from ai.ml.knn.model_selection_utils import (
    backward_elimination_with_cross_validation,
    postprocess_cv_results
)


def model_selection_step(
        bucket_name: str,
        run_timestamp: str,
        base_data_dir: str,
        raw_data_prefix: str,
        model_selection_dir: str,
        logs_dir: str,
        table_columns: Dict[str, List[str]],
        table_categorical_features: Dict[str, List[str]],
        table_numerical_features: Dict[str, List[str]],
        table_target_variables: Dict[str, str],
        table_date_columns: Dict[str, str],
        table_filter_low_frequency_class_counts: Dict[str, bool],
        table_must_have_features: Dict[str, List[str]],
        random_seed: int = 42,
        specific_tenant_id: str = None,
) -> None:
    """
    Performs a data-driven model selection workflow that includes data collection, preprocessing,
    feature selection, hyperparameter optimization, and result registration for multiple datasets
    and tenants. The function processes data by iterating over tenant-specific datasets within tables
    and applies backward elimination combined with cross-validation to select optimal model configurations.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket where data and logs will be registered.
    run_timestamp : str
        A unique timestamp string used to identify the specific run of the process.
    base_data_dir : str
        The base local directory where required data and logs are stored.
    raw_data_prefix : str
        Prefix to locate the raw data files within the base data directory.
    model_selection_dir : str
        Directory path to save the results of the model selection process.
    logs_dir : str
        Directory path for saving process execution logs.
    table_columns : Dict[str, List[str]]
        Dictionary mapping table names to their respective lists of all columns.
    table_categorical_features : Dict[str, List[str]]
        Dictionary mapping table names to their respective lists of categorical feature columns.
    table_numerical_features : Dict[str, List[str]]
        Dictionary mapping table names to their respective lists of numerical feature columns.
    table_target_variables : Dict[str, str]
        Dictionary mapping table names to their respective target variable columns.
    table_date_columns : Dict[str, str]
        Dictionary mapping table names to their respective date columns.
    table_filter_low_frequency_class_counts : Dict[str, bool]
        Dictionary mapping table names to a flag indicating whether to filter low-frequency classes.
    table_must_have_features : Dict[str, List[str]]
        Dictionary mapping table names to a list of features that must be present in the feature selection.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility (default is 42).

    Returns
    -------
    None
        This function does not return any value. All outputs are stored locally and registered with S3.
    """
    absolute_logs_dir = os.path.join(base_data_dir, logs_dir, 'model_selection')
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
    tenant_ids = fetch_tenant_ids(specific_tenant_id=specific_tenant_id)
    for tenant_id in tenant_ids:
        data_collection_step(raw_dataset_dir=raw_dataset_dir,
                             tenant_id=tenant_id)

    table_names = [item for item in os.listdir(raw_dataset_dir) if not item.startswith('.')]
    # finding tables of each tenant id
    tenant_tables = {}
    for table_name in table_names:
        raw_table_dir = os.path.join(raw_dataset_dir, table_name)
        table_tenant_ids = [file_name.split('.')[0] for file_name in os.listdir(raw_table_dir)
                            if not file_name.startswith('.')]
        for t_id in table_tenant_ids:
            if t_id not in tenant_tables:
                tenant_tables[t_id] = [table_name]
            else:
                tenant_tables[t_id].append(table_name)

    # Process each tenant first
    model_selection_results = {}
    artifact_dirs = []
    artifact_prefixes = []
    for tenant_id in tenant_ids:
        logger.info("========== START BE PROCESSING TENANT ID: %s ==========", tenant_id)

        # Process each table for this tenant
        tenant_artifact_dirs = []
        tenant_artifact_prefixes = []
        for table_name in tenant_tables[tenant_id]:
            logger.info(f"Processing table {table_name} for tenant {tenant_id}")

            # Configurations for Tenant-Table Model Selection
            TENANT_MODEL_SELECTION_PREFIX = os.path.join(model_selection_dir,
                                                         tenant_id,
                                                         run_timestamp,
                                                         table_name)
            TENANT_MODEL_SELECTION_DIR = os.path.join(base_data_dir,
                                                      TENANT_MODEL_SELECTION_PREFIX)
            os.makedirs(TENANT_MODEL_SELECTION_DIR, exist_ok=True)
            logger.info("Model selection directory created at: %s", TENANT_MODEL_SELECTION_DIR)
            N_NEIGHBORS_OPTIONS = [3, 5, 7, 9]
            WEIGHTS_OPTIONS = ['uniform', 'distance']
            accuracy_tolerance = 0.05
            n_jobs = -1
            early_stopping_iterations = 10
            if table_name not in model_selection_results:
                model_selection_results[table_name] = {}

            # ======== Tenant-Table Data Loading ========
            logger.info(f"Loading the data for table {table_name} and tenant {tenant_id}")
            raw_table_dir = os.path.join(raw_dataset_dir, table_name)
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
            )

            # ======== Preprocessing ========
            logger.info(f"Preprocessing the data for table {table_name} and tenant {tenant_id}")
            df = table_data['raw_df'].copy()
            output = general_preprocessing(
                df=df,
                target_variable=table_data['target_variable'],
                numerical_features=table_data['numerical_features'],
                categorical_features=table_data['categorical_features'],
                filter_low_frequency_class_counts=table_data['filter_low_frequency_class_counts'],
                random_seed=random_seed,
                logger=logger
            )
            df = output["df"]
            if df.empty:
                logger.info(f"No data for table {table_name} for tenant {tenant_id} after preprocessing,"
                            f" skipping.")
                continue
            numerical_features = output["numerical_features"]
            categorical_features = output["categorical_features"]
            constant_columns = output["constant_columns"]
            removed_classes = output["removed_classes"]
            class_counts = output["class_counts"]
            if table_data['target_variable'] in constant_columns:
                logger.warning(f"Target variable is constant for tenant {tenant_id}, skipping table {table_name}.")
                continue

            min_required_samples = 5 * len(df[table_data['target_variable']].unique())
            if len(df) < min_required_samples:
                logger.warning(f"Insufficient samples for table {table_name}, tenant {tenant_id}: "
                               f" Need {min_required_samples}, have {len(df)}.")
                continue
            max_k = 9
            cv_folds = 5
            min_samples_per_fold = max_k + 1
            min_total_samples_knn = min_samples_per_fold * cv_folds / (cv_folds - 1)
            if len(df) < min_total_samples_knn:
                logger.warning(
                    f"Insufficient samples for KNN with k={max_k} for table {table_name}, tenant {tenant_id}: "
                    f"Need at least {int(min_total_samples_knn)}, have {len(df)}.")
                continue
            num_samples_for_efficiency = 10_000
            if num_samples_for_efficiency < min_required_samples:
                logger.warning(
                    f"Cannot sample {num_samples_for_efficiency} samples with "
                    f"min_per_class=5 for {len(df[table_data['target_variable']].unique())} classes "
                    f"in table {table_name}, tenant {tenant_id}: Need at least {min_required_samples} samples.")
                continue

            # Validate must_have_features exist in the dataset
            must_have_features = table_must_have_features[table_name]
            all_features = numerical_features + categorical_features
            invalid_must_have = [f for f in must_have_features if f not in all_features]
            if invalid_must_have:
                raise ValueError(f"Must-have features not found in dataset: {invalid_must_have}")

            # ======== Backward Elimination Procedure with CV-based Evaluation ========
            logger.info(f"Starting backward-elimination process for table {table_name} and tenant {tenant_id}")
            backward_elimination_info = backward_elimination_with_cross_validation(
                df=df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                target_variable=table_data['target_variable'],
                must_have_features=must_have_features,
                n_neighbors_option=N_NEIGHBORS_OPTIONS,
                weights_options=WEIGHTS_OPTIONS,
                accuracy_tolerance=accuracy_tolerance,
                n_jobs=n_jobs,
                early_stopping_iterations=early_stopping_iterations,
                sampling_for_efficiency=True,
                num_samples_for_efficiency=num_samples_for_efficiency,
                random_seed=random_seed,
                logger=logger
            )
            cv_results = backward_elimination_info['cv_results']

            # post-process cv_results
            cv_results, cv_best_minimal_config = postprocess_cv_results(cv_results=cv_results)

            # Create hyperparameters for this tenant
            tenant_hyperparameters = {
                "run_timestamp": run_timestamp,
                "version": run_timestamp,
                "table_name": table_name,
                "tenant_id": tenant_id,
                "model_name": "sklearn.neighbors.KNeighborsClassifier",
                "raw_data_path": table_data['csv_path'],
                "model_selection_dir": TENANT_MODEL_SELECTION_DIR,
                "random_seed": random_seed,
                "target_variable": table_data['target_variable'],
                "class_counts": class_counts.to_dict(),
                "removed_classes": removed_classes,
                "filter_low_frequency_class_counts": table_data['filter_low_frequency_class_counts'],
                "num_samples": len(df),
                "n_neighbors_options": N_NEIGHBORS_OPTIONS,
                "weights_options": WEIGHTS_OPTIONS,
                'backward_elimination_accuracy_tolerance': accuracy_tolerance,
            }

            # register to local storage for this tenant
            register_model_selection_to_local_storage(
                df=df,
                cv_results=cv_results,
                cv_best_minimal_config=cv_best_minimal_config,
                hyperparameters=tenant_hyperparameters,
                tenant_model_selection_dir=TENANT_MODEL_SELECTION_DIR,
                logger=logger
            )

            model_selection_results[table_name][tenant_id] = {
                "df": df,
                "cv_results": cv_results,
                "cv_best_minimal_config": cv_best_minimal_config,
                "hyperparameters": tenant_hyperparameters
            }

            # Log results for this tenant
            logger.info(f"==== Backward Elimination Summary: "
                        f"Table: {table_name} | "
                        f"Target Variable: {table_data['target_variable']} | "
                        f"Tenant ID {tenant_id} ====")
            logger.info("Best Features:")
            logger.info(f"{cv_best_minimal_config['features']}")
            logger.info(f"Best N Neighbors: {cv_best_minimal_config['n_neighbors']}")
            logger.info(f"Best Weight: {cv_best_minimal_config['weights']}")
            logger.info("=============================================")

            # Track artifact directories and prefixes for this tenant
            tenant_artifact_dirs.append(TENANT_MODEL_SELECTION_DIR)
            tenant_artifact_prefixes.append(TENANT_MODEL_SELECTION_PREFIX)
        artifact_dirs.append(tenant_artifact_dirs)
        artifact_prefixes.append(tenant_artifact_prefixes)

    # Register this tenant's results to S3 after processing all tables
    logger.info("Registering model selection artifacts into S3... ")
    register_data_to_s3(
        bucket_name=bucket_name,
        run_timestamp=run_timestamp,
        raw_dataset_dir=raw_dataset_dir,
        raw_dataset_prefix=raw_data_prefix,
        artifact_dirs=artifact_dirs,
        artifact_prefixes=artifact_prefixes,
        logger=logger
    )

    logger.info("Completed model selection for all tenants")
