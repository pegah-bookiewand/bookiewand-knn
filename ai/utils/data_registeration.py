import os
import logging
import json
import pickle
from typing import List, Dict, Union, Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from ai.utils.initialization import get_s3_client
from ai.utils.s3_utils import create_bucket, upload_directory_to_s3
from ai.agents.jouranl_attr_classifier.data_types.general import (
    LLMJournalLinePredictionResponse,
)


def register_data_to_local_storage(
        raw_df: pd.DataFrame,
        train_df: pd.DataFrame,
        encoders: Dict[str, Union[StandardScaler, OneHotEncoder]],
        hyperparameters: Dict[str, Union[float, int, str]],
        model: KNeighborsClassifier,
        tenant_checkpoint_dir: str,
        logger: logging.Logger
) -> None:
    """
    Register training data, model, encoders, and hyperparameters to local storage.

    This function saves the following items to the specified directory:
    1. Hyperparameters as a JSON file
    2. Training DataFrame as a CSV file
    3. Encoders as a pickle file
    4. Trained model as a pickle file

    Parameters
    ----------
    raw_df: pd.DataFrame
        The raw data used for training.
    train_df : pd.DataFrame
        The final training data to be saved as a CSV file.
    encoders : dict of {str: Union[StandardScaler, OneHotEncoder]}
        The encoders to be saved as a pickle file.
    hyperparameters : dict of {str: Union[float, int, str]}
        The hyperparameters to be saved as a JSON file.
    model : KNeighborsClassifier
        The trained model to be saved as a pickle file.
    tenant_checkpoint_dir : str
        The directory where the files will be saved.
    logger: logging.Logger

    Returns
    -------
    None
        This function does not return anything. It saves the data to disk.
    """
    logger.info("Starting the process of saving data to local storage.")

    # Save hyperparameters
    hyperparams_path = os.path.join(tenant_checkpoint_dir, "hyperparameters.json")
    logger.info("Saving hyperparameters to %s", hyperparams_path)
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    logger.info("Saved hyperparameters to %s", hyperparams_path)

    # Save the raw and final training data
    raw_df_path = os.path.join(tenant_checkpoint_dir, "raw.csv")
    logger.info("Saving final training data to %s", raw_df_path)
    raw_df.to_csv(raw_df_path, index=False)
    logger.info("Saved raw data to %s", raw_df_path)

    final_training_df_path = os.path.join(tenant_checkpoint_dir, "train.csv")
    logger.info("Saving final training data to %s", final_training_df_path)
    train_df.to_csv(final_training_df_path, index=False)
    logger.info("Saved final training data to %s", final_training_df_path)

    # Save encoders
    encoders_path = os.path.join(tenant_checkpoint_dir, "encoders.pkl")
    logger.info("Saving encoders to %s", encoders_path)
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info("Saved encoders to %s", encoders_path)

    # Save trained model
    model_path = os.path.join(tenant_checkpoint_dir, "model.pkl")
    logger.info("Saving trained model to %s", model_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved final model to %s", model_path)

    logger.info("Data successfully saved to local storage.")


def register_tfidf_data_to_local_storage(
        raw_df: pd.DataFrame,
        train_df: pd.DataFrame,
        feature_tokenizer: Dict[str, TfidfVectorizer],
        feature_token_matrix: Dict[str, csr_matrix],
        hyperparameters: Dict[str, Union[float, int, str]],
        tenant_checkpoint_dir: str,
        logger: logging.Logger
) -> None:
    logger.info("Starting the process of saving data to local storage.")

    # Save hyperparameters
    hyperparams_path = os.path.join(tenant_checkpoint_dir, "hyperparameters.json")
    logger.info("Saving hyperparameters to %s", hyperparams_path)
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    logger.info("Saved hyperparameters to %s", hyperparams_path)

    # Save the raw and final training data
    raw_df_path = os.path.join(tenant_checkpoint_dir, "raw.csv")
    logger.info("Saving final training data to %s", raw_df_path)
    raw_df.to_csv(raw_df_path, index=False)
    logger.info("Saved raw data to %s", raw_df_path)

    final_training_df_path = os.path.join(tenant_checkpoint_dir, "train.csv")
    logger.info("Saving final training data to %s", final_training_df_path)
    train_df.to_csv(final_training_df_path, index=False)
    logger.info("Saved final training data to %s", final_training_df_path)

    # Save encoders
    feature_tokenizer_path = os.path.join(tenant_checkpoint_dir, "feature_tokenizer.pkl")
    logger.info("Saving feature_tokenizer to %s", feature_tokenizer_path)
    with open(feature_tokenizer_path, "wb") as f:
        pickle.dump(feature_tokenizer, f)
    logger.info("Saved feature_tokenizer to %s", feature_tokenizer_path)

    # Save trained model
    feature_token_matrix_path = os.path.join(tenant_checkpoint_dir, "feature_token_matrix.pkl")
    logger.info("Saving feature_token_matrix to %s", feature_token_matrix_path)
    with open(feature_token_matrix_path, "wb") as f:
        pickle.dump(feature_token_matrix, f)
    logger.info("Saved feature_token_matrix to %s", feature_token_matrix_path)

    logger.info("Data successfully saved to local storage.")


def register_model_selection_to_local_storage(
        df: pd.DataFrame,
        cv_results: List[Dict],
        cv_best_minimal_config: Dict,
        hyperparameters: Dict[str, Union[float, int, str, Dict[str, int], List[int], List[str]]],
        tenant_model_selection_dir: str,
        logger: logging.Logger,
) -> None:
    """
    Registers the results and configuration of a model selection process into local storage.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe used during the model selection process.
    cv_results : list
        A list of dictionaries containing cross-validation results for different configurations.
    cv_best_minimal_config : dict
        A dictionary storing the best configuration details according to cross-validation.
    hyperparameters : dict
        The hyperparameters and metadata associated with the model selection process.
        Keys include strings, numbers, nested dictionaries, or lists.
    tenant_model_selection_dir : str
        The directory path where the model selection artifacts will be saved.
    logger: logging.Logger

    Returns
    -------
    None
        This function does not return anything. Artifacts are saved to the specified directory.
    """
    logger.info("Starting the process of saving model selection artifacts to local storage.")

    # Save hyperparameters
    hyperparams_path = os.path.join(tenant_model_selection_dir, "hyperparameters.json")
    logger.info("Saving hyperparameters to %s", hyperparams_path)
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    logger.info("Saved hyperparameters to %s", hyperparams_path)

    # Saving cv_results
    cv_results_path = os.path.join(tenant_model_selection_dir, "cv_results.json")
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=4)

    logger.info("Saved cv_results to %s", cv_results_path)
    cv_results_txt_path = os.path.join(tenant_model_selection_dir, "cv_results.txt")
    with open(cv_results_txt_path, 'w') as file:
        for index, item in enumerate(cv_results, start=1):
            line = (
                f"{index}. Features: {item['features']} | "
                f"Mean Accuracy: {item['mean_accuracy']} | "
                f"N Neighbors: {item['n_neighbors']} | "
                f"Weight: {item['weights']}\n"
            )
            file.write(line)
    logger.info("Saved cv_results.txt to %s", cv_results_txt_path)

    # Saving cv_best_minimal_config
    cv_best_minimal_config_path = os.path.join(tenant_model_selection_dir, "cv_best_minimal_config.json")
    with open(cv_best_minimal_config_path, "w") as f:
        json.dump(cv_best_minimal_config, f, indent=4)
    logger.info("Saved hyperparameters to %s", cv_best_minimal_config)

    # Saving dataframe used in model selection
    df_path = os.path.join(tenant_model_selection_dir, "df.csv")
    df.to_csv(df_path, index=False)
    logger.info("Saved df data to %s", df_path)

    logger.info("Artifacts successfully saved to local storage.")


def register_evaluation_data_to_local_storage(
        test_df: pd.DataFrame,
        task_eval_df: pd.DataFrame,
        response: LLMJournalLinePredictionResponse,
        result_dict: Dict[str, Any],
        confusion_matrix: pd.DataFrame,
        metrics_df: pd.DataFrame,
        evaluation_dir: str,
        logger: logging.Logger,
) -> None:
    """
    Save evaluation data components to local storage directory.

    Creates the evaluation directory if it doesn't exist and saves all evaluation
    artifacts including test data, task evaluation data, model response, confusion
    matrix, and metrics to separate files.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataset used for evaluation.
    task_eval_df : pd.DataFrame
        Task-specific evaluation results dataset.
    response : LLMJournalLinePredictionResponse
        Model prediction response containing results and hyperparameters.
    result_dict : Dict[str, Any]
        A dictionary containing the results for report generation purposes.
    confusion_matrix : pd.DataFrame
        Confusion matrix from model evaluation.
    metrics_df : pd.DataFrame
        Evaluation metrics dataset.
    evaluation_dir : str
        Local directory path where evaluation files will be saved.
    logger : logging.Logger
        Logger instance for tracking the save process.

    Returns
    -------
    None
    """

    logger.info("Starting the process of saving evaluation data to local storage.")
    logger.info("Saving test data to %s", evaluation_dir)
    os.makedirs(evaluation_dir, exist_ok=True)

    # response
    response_path = os.path.join(evaluation_dir, "response.json")
    logger.info("Saving response to %s", response_path)
    with open(response_path, "w") as f:
        json.dump(response.model_dump(), f, indent=4)

    # response
    results_path = os.path.join(evaluation_dir, "results.json")
    logger.info("Saving results JSON to %s", results_path)
    with open(results_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    # test_df
    test_df_path = os.path.join(evaluation_dir, "test_df.csv")
    logger.info("Saving test_df to %s", test_df_path)
    test_df.to_csv(test_df_path, index=False)

    # task_eval_df
    task_eval_df_path = os.path.join(evaluation_dir, "task_eval_df.csv")
    logger.info("Saving task_eval_df to %s", task_eval_df_path)
    task_eval_df.to_csv(task_eval_df_path, index=False)

    # confusion_matrix
    confusion_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.csv")
    logger.info("Saving confusion_matrix to %s", confusion_matrix_path)
    confusion_matrix.to_csv(confusion_matrix_path, index=False)

    # metrics_df
    metrics_df_path = os.path.join(evaluation_dir, "metrics_df.csv")
    logger.info("Saving metrics_df to %s", metrics_df_path)
    metrics_df.to_csv(metrics_df_path, index=False)

    logger.info("Data successfully saved to local storage.")


def register_data_to_s3(
        bucket_name: str,
        run_timestamp: str,
        raw_dataset_dir: str,
        raw_dataset_prefix: str,
        artifact_dirs: List[List[str]],
        artifact_prefixes: List[List[str]],
        logger: logging.Logger
) -> None:
    """
    Uploads raw dataset and artifacts to the S3 bucket.

    Parameters
    ----------
    bucket_name: str
    run_timestamp : str
        The timestamp associated with the current run for organizing datasets.
    raw_dataset_dir : str
        The local directory containing the raw dataset to upload.
    raw_dataset_prefix : str
        The prefix under which the raw dataset should be stored in S3.
    artifact_dirs : List[List[str]]
        List of local directories containing artifacts to be uploaded for each table, and each tenant_id.
    artifact_prefixes : List[List[str]]
        Corresponding list of prefixes for storing artifacts in S3 for each table, and each tenant_id.
    logger: logging.Logger

    Returns
    -------
    None
    """
    logger.info("Initializing S3 client.")
    s3_client = get_s3_client()

    logger.info(f"Ensuring S3 bucket '{bucket_name}' exists.")
    create_bucket(s3_client=s3_client, bucket_name=bucket_name)

    logger.info(f"Uploading raw dataset from {raw_dataset_dir} to S3 at {raw_dataset_prefix}/{run_timestamp}.")
    upload_directory_to_s3(s3_client=s3_client,
                           local_directory=raw_dataset_dir,
                           bucket_name=bucket_name,
                           prefix=f"{raw_dataset_prefix}/{run_timestamp}")

    for artifact_dir, artifact_prefix in zip(artifact_dirs, artifact_prefixes):
        logger.info(f"Uploading artifacts from {artifact_dir} to S3 at {artifact_prefix}.")
        for idx, table_artifact_dir in enumerate(artifact_dir):
            table_artifact_prefix = artifact_prefix[idx]
            upload_directory_to_s3(s3_client=s3_client,
                                   local_directory=table_artifact_dir,
                                   bucket_name=bucket_name,
                                   prefix=table_artifact_prefix)

    logger.info("S3 upload process completed successfully.")
