import os
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score
)

from ai.utils.initialization import get_s3_client
from ai.utils.s3_utils import list_s3_directories
from ai.utils.data_collection import (
    fetch_table_data,
    llm_context_collection_step,
    model_collection_step,
    fetch_table_data_by_line_item_ids
)
from ai.utils.data_types import JournalLinePredictionResponse
from ai.ml.knn.inference import knn_inference
from ai.agents.jouranl_attr_classifier.inference import llm_inference
from ai.agents.utils.initialization import get_llm
from ai.utils.data_registeration import register_evaluation_data_to_local_storage
from ai.utils.s3_utils import upload_directory_to_s3


async def evaluation_step(
        task_name: str,
        method_name: str,
        method_run_timestamp: str,
        base_data_dir: str,
        bucket_name: str,
        checkpoints_prefix: str,
        logger: logging.Logger
) -> None:
    """
    Performs evaluation on trained models for specified tasks and methods.

    This function validates trained models or contexts by running inference on test data
    and computing classification metrics. It supports both KNN and LLM methods for
    GST validation and account mapping validation tasks.

    Parameters
    ----------
    task_name : str
        The validation task to perform. Must be one of ['GST_VALIDATION', 'ACCOUNT_MAPPING_VALIDATION'].
    method_name : str
        The method to evaluate. Must be one of ['KNN', 'LLM'].
    method_run_timestamp : str
        The timestamp identifier for the specific model/context version to evaluate.
    base_data_dir : str
        The local base directory path for storing evaluation data and artifacts.
    bucket_name : str
        The name of the S3 bucket containing model checkpoints and data.
    checkpoints_prefix : str
        The S3 prefix path where model checkpoints and contexts are stored.
    logger : logging.Logger
        Logger instance for recording evaluation progress and results.

    Returns
    -------
    None
        This function performs evaluation and saves results to local storage and S3,
        but does not return any values.
    """
    logger.info(f"Starting evaluation step - "
                f"Task: {task_name}, "
                f"Method: {method_name}, "
                f"Timestamp: {method_run_timestamp}")

    assert task_name in ['GST_VALIDATION', 'ACCOUNT_MAPPING_VALIDATION'], \
        ("Invalid task_name argument. "
         "Valid task_name arguments are in ['GST_VALIDATION', 'ACCOUNT_MAPPING_VALIDATION']")
    assert method_name in ['KNN', 'LLM'], \
        ("Invalid method_name argument. "
         "Valid method_name arguments are in ['knn', 'llm'].")
    s3_client = get_s3_client()

    # read all tenant ids that have prepared models
    logger.info(f"Discovering tenant IDs from S3 bucket: {bucket_name}, prefix: {checkpoints_prefix}")
    tenant_ids = list_s3_directories(s3_client=s3_client,
                                     bucket_name=bucket_name,
                                     prefix=checkpoints_prefix)
    if not tenant_ids:
        raise ValueError(f"There are no trained models or preprocessed context.")

    logger.info(f"Found {len(tenant_ids)} tenant IDs: {tenant_ids}")

    # Filter by SPECIFIC_TENANT_ID if set in environment
    specific_tenant_id = os.getenv('SPECIFIC_TENANT_ID', '').strip()
    if specific_tenant_id:
        logger.info(f"Filtering for specific tenant ID: {specific_tenant_id}")
        if specific_tenant_id in tenant_ids:
            tenant_ids = [specific_tenant_id]
            logger.info(f"Filtered to single tenant: {specific_tenant_id}")
        else:
            logger.error(f"Specified tenant ID {specific_tenant_id} not found in available tenants: {tenant_ids}")
            raise ValueError(f"SPECIFIC_TENANT_ID '{specific_tenant_id}' not found in available tenant IDs")
    else:
        logger.info("No SPECIFIC_TENANT_ID set, processing all tenants")

    # extract tenant ids that have the given method_run_timestamp
    valid_tenant_ids = []
    logger.info(f"Validating tenant IDs for method run timestamp: {method_run_timestamp}")
    for tenant_id in tenant_ids:
        tenant_method_run_timestamp = list_s3_directories(s3_client=s3_client,
                                                          bucket_name=bucket_name,
                                                          prefix=os.path.join(checkpoints_prefix, tenant_id))
        if method_run_timestamp in tenant_method_run_timestamp:
            valid_tenant_ids.append(tenant_id)
            logger.info(f"Tenant {tenant_id} has valid timestamp {method_run_timestamp}")
        else:
            logger.error(
                f"Tenant {tenant_id} missing timestamp {method_run_timestamp}. Available: {tenant_method_run_timestamp}")
            raise ValueError(f"Tenant id {tenant_id} does not have the {method_name} method "
                             f"with run timestamp {method_run_timestamp}.")

    logger.info(f"Validated {len(valid_tenant_ids)} tenant IDs for evaluation: {valid_tenant_ids}")

    # load the method data
    logger.info(f"Loading {method_name} method data for timestamp {method_run_timestamp}")
    if method_name == "LLM":
        method_data = llm_context_collection_step(
            base_data_dir=base_data_dir,
            bucket_name=bucket_name,
            checkpoints_prefix=checkpoints_prefix,
            checkpoints_dir=checkpoints_prefix,
            logger=logger,
            context_version=method_run_timestamp
        )
        method_key = "context"
        logger.info("LLM context data loaded successfully.")
    elif method_name == "KNN":
        method_data = model_collection_step(base_data_dir=base_data_dir,
                                            bucket_name=bucket_name,
                                            checkpoints_prefix=checkpoints_prefix,
                                            checkpoints_dir=checkpoints_prefix,
                                            logger=logger,
                                            model_version=method_run_timestamp)
        method_key = "models"
        logger.info("KNN model data loaded successfully.")
    else:
        raise NotImplementedError(f"Invalid method name: {method_name}.")

    # loading the test data
    logger.info("Loading test data from 'gst_account_validation_eval_set' table")
    test_df = fetch_table_data(table_name='gst_account_validation_eval_set',
                               logger=logger)
    logger.info(f"Loaded test data with {len(test_df)} records")

    # evaluation
    logger.info(f"Starting evaluation loop for {len(valid_tenant_ids)} tenants")
    for tenant_idx, tenant_id in enumerate(valid_tenant_ids):
        logger.info(f"Processing tenant {tenant_idx + 1}/{len(valid_tenant_ids)}: {tenant_id}")

        tenant_method_prefix = str(os.path.join(checkpoints_prefix, tenant_id, method_run_timestamp))
        tenant_table_names = list_s3_directories(s3_client=s3_client,
                                                 bucket_name=bucket_name,
                                                 prefix=tenant_method_prefix)
        logger.info(f"Tenant {tenant_id} has {len(tenant_table_names)} tables: {tenant_table_names}")

        for table_idx, table_name in enumerate(tenant_table_names):
            logger.info(f"Processing table {table_idx + 1}/{len(tenant_table_names)} "
                        f"for tenant {tenant_id}: {table_name}")

            task_tenant_table_eval_df = test_df[(test_df['task_name'] == task_name) &
                                                (test_df['tenant_id'] == tenant_id) &
                                                (test_df['table_name'] == table_name)].reset_index(drop=True)
            if len(task_tenant_table_eval_df) == 0:
                logger.info(f"No test data found for task {task_name}, "
                            f"tenant id {tenant_id}, and, table {table_name}.")
                continue

            line_item_id_list = task_tenant_table_eval_df['line_item_id'].unique().tolist()
            logger.info(f"Found {len(task_tenant_table_eval_df)} evaluation records, "
                        f"using {len(line_item_id_list)} unique line items")

            # table_names = list_database_tables_with_columns(logger=logger)
            logger.info(f"Fetching table data for {len(line_item_id_list)} line items from table {table_name}")
            tenant_table_test_df = fetch_table_data_by_line_item_ids(
                table_name=table_name,
                tenant_id=tenant_id,
                line_item_id_list=line_item_id_list,
                logger=logger
            )
            logger.info(f"Retrieved {len(tenant_table_test_df)} records from table {table_name}")

            tenant_table_method_data = method_data[method_key][tenant_id][table_name]
            logger.info(f"Loaded method data for tenant {tenant_id}, table {table_name}")

            # inference
            logger.info(f"Starting {method_name} inference for tenant {tenant_id}, table {table_name}")
            if method_name == "LLM":
                try:
                    # loading the OpenAI client
                    logger.info("Initializing OpenAI client for LLM inference")
                    openai_client, deterministic_settings, chat_client_info = get_llm(logger=logger)
                    response = await llm_inference(
                        tenant_id=tenant_id,
                        table_name=table_name,
                        input_data=tenant_table_test_df,
                        context=tenant_table_method_data,
                        client=openai_client,
                        chat_client_info=chat_client_info,
                        model_settings=deterministic_settings,
                        logger=logger
                    )
                    logger.info(f"LLM inference completed successfully. "
                                f"Generated {len(response.results)} predictions.")
                    if not response.results:
                        logger.warning(f"No results generated for tenant {tenant_id}, table {table_name}. Skipping evaluation.")
                        continue
                except Exception as e:
                    logger.error(f"LLM inference failed for tenant {tenant_id}, table {table_name}: {str(e)}")
                    raise ValueError(f"LLM inference failed for tenant id {tenant_id}, {table_name}. "
                                     f"{str(e)}")
            elif method_name == "KNN":
                logger.info("KNN inference not yet implemented")
                response = knn_inference(tenant_id=tenant_id,
                                         table_name=table_name,
                                         df=tenant_table_test_df,
                                         model_data=tenant_table_method_data)
            else:
                raise NotImplementedError(f"Invalid method name: {method_name}.")

            # classification scores
            logger.info(f"Computing classification metrics for tenant {tenant_id}, table {table_name}")
            tenant_table_results = {
                'line_item_id': [],
                'true': [],
                'pred': [],
            }
            for result in response.results:
                result_line_item_id = result.record.root.line_item_id
                result_true = task_tenant_table_eval_df[
                    task_tenant_table_eval_df['line_item_id'] == result_line_item_id
                    ]['correct_target_variable'].item()
                result_prediction = result.target_prediction
                tenant_table_results['line_item_id'].append(result_line_item_id)
                tenant_table_results['true'].append(result_true)
                tenant_table_results['pred'].append(result_prediction)

            tenant_table_results = pd.DataFrame.from_dict(tenant_table_results)
            logger.info(f"Prepared results dataframe with {len(tenant_table_results)} prediction results")

            tenant_table_classification_scores = evaluate_classification_results(
                tenant_table_results=tenant_table_results)

            # Extract key metrics for logging
            metrics_df = tenant_table_classification_scores['metrics_df']
            accuracy = metrics_df[metrics_df['Metric'] == 'Accuracy']['Value'].iloc[0]
            f1_score = metrics_df[metrics_df['Metric'] == 'F1-Score (Weighted)']['Value'].iloc[0]
            logger.info(f"Classification metrics computed - Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")

            # evaluation JSON report for report generation scripts
            tenant_table_report = {
                "metrics": {},
                "records": []
            }
            for metric_idx, metric_row in metrics_df.iterrows():
                metric_name = metric_row['Metric']
                metric_value = metric_row['Value']
                tenant_table_report['metrics'][metric_name] = metric_value
            for result in response.results:
                result_line_item_id = result.record.root.line_item_id
                result_true = task_tenant_table_eval_df[
                    task_tenant_table_eval_df['line_item_id'] == result_line_item_id
                    ]['correct_target_variable'].item()

                if method_name == "LLM":
                    result_explanations = []
                    for explanation in result.explanations:
                        expl = {
                            "similar_feature": explanation.feature,
                            "distance": explanation.distance,
                            "record": explanation.record.model_dump(),
                        }
                        result_explanations.append(expl)
                    result_json = {
                        "true": result_true,
                        "pred": result.target_prediction,
                        "confidence_score": result.confidence_score,
                        "reasoning": result.reasoning,
                        "record": result.record.model_dump(),
                        "explanations": result_explanations,
                    }
                elif method_name == "KNN":
                    result_explanations = []
                    for explanation in result.explanations:
                        expl = {
                            "distance": explanation.distance,
                            "record": explanation.record.model_dump(),
                        }
                        result_explanations.append(expl)
                    result_json = {
                        "true": result_true,
                        "pred": result.target_prediction,
                        "record": result.record.model_dump(),
                        "explanations": result_explanations,
                    }
                else:
                    raise NotImplementedError(f"Invalid method name: {method_name}.")
                tenant_table_report['records'].append(result_json)

            # registering evaluation data to local storage
            evaluation_dir = os.path.join(base_data_dir,
                                          checkpoints_prefix,
                                          tenant_id,
                                          method_run_timestamp,
                                          table_name,
                                          'evaluation')
            logger.info(f"Saving evaluation data to local storage: {evaluation_dir}")
            register_evaluation_data_to_local_storage(
                test_df=tenant_table_test_df,
                task_eval_df=task_tenant_table_eval_df,
                response=response,
                result_dict=tenant_table_report,
                confusion_matrix=tenant_table_classification_scores['confusion_matrix_df'],
                metrics_df=tenant_table_classification_scores['metrics_df'],
                evaluation_dir=evaluation_dir,
                logger=logger
            )

            # registering evaluation data to S3
            evaluation_prefix = os.path.join(checkpoints_prefix,
                                             tenant_id,
                                             method_run_timestamp,
                                             table_name,
                                             'evaluation')
            logger.info(f"Uploading evaluation data to S3: {bucket_name}/{evaluation_prefix}")
            upload_directory_to_s3(s3_client=s3_client,
                                   local_directory=evaluation_dir,
                                   bucket_name=bucket_name,
                                   prefix=evaluation_prefix)
            logger.info(f"Successfully uploaded evaluation data to S3")
            logger.info(f"Completed evaluation for tenant {tenant_id}, table {table_name}")
    logger.info(f"Evaluation step completed successfully.")


def evaluate_classification_results(
        tenant_table_results: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate classification results and generate performance metrics.

    This function computes comprehensive classification metrics and creates a confusion matrix
    for binary or multi-class classification problems. It automatically determines the most
    appropriate recommended score based on the number of classes.

    Parameters
    ----------
    tenant_table_results : pd.DataFrame
        DataFrame containing classification results with 'true' and 'pred' columns.
        'true' column contains actual/ground truth labels.
        'pred' column contains predicted labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation results with two keys:
        - 'metrics_df' : pd.DataFrame
            DataFrame with classification metrics including Accuracy, Balanced Accuracy,
            Precision (Weighted), Recall (Weighted), F1-Score (Weighted), Matthews
            Correlation, and Recommended Score. Numeric values are rounded to 4 decimals.
        - 'confusion_matrix_df' : pd.DataFrame
            Confusion matrix as DataFrame with rows labeled as 'True_{label}' and
            columns labeled as 'Pred_{label}' for all unique labels in the data.
    """

    y_true = tenant_table_results['true'].values
    y_pred = tenant_table_results['pred'].values

    # Get all unique labels from both true and predicted
    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Get only labels that appear in y_true for metrics calculation
    true_labels = np.unique(y_true)

    # Determine best single score
    is_binary = len(true_labels) == 2
    if is_binary:
        recommended_score = 'Matthews Correlation'
    else:
        recommended_score = 'F1-Score (Weighted)'

    # Essential metrics only - use labels parameter to avoid warnings
    metrics_data = [
        {
            'Metric': 'Accuracy',
            'Value': accuracy_score(y_true, y_pred)
        },
        {
            'Metric': 'Balanced Accuracy',
            'Value': balanced_accuracy_score(y_true, y_pred)
        },
        {
            'Metric': 'Precision (Weighted)',
            'Value': precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=true_labels)
        },
        {
            'Metric': 'Recall (Weighted)',
            'Value': recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=true_labels)
        },
        {
            'Metric': 'F1-Score (Weighted)',
            'Value': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=true_labels)
        },
        {
            'Metric': 'Matthews Correlation',
            'Value': matthews_corrcoef(y_true, y_pred)
        },
        {
            'Metric': 'Recommended Score',
            'Value': recommended_score
        }
    ]

    metrics_df = pd.DataFrame(metrics_data)

    # Round only numeric values, leave strings as-is
    numeric_mask = pd.to_numeric(metrics_df['Value'], errors='coerce').notna()
    metrics_df.loc[numeric_mask, 'Value'] = pd.to_numeric(metrics_df.loc[numeric_mask, 'Value']).round(4)

    # Confusion matrix - use all labels to show full picture including wrong predictions
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{label}' for label in labels],
        columns=[f'Pred_{label}' for label in labels]
    )

    return {
        'metrics_df': metrics_df,
        'confusion_matrix_df': cm_df
    }


def evaluate_knn_response(
        response: JournalLinePredictionResponse,
) -> float:
    results = {
        "prediction": [],
        "ground_truth": [],
        "is_error": [],
        "class_frequency": [],
        "min_distance": [],
        "max_distance": [],
        "avg_distance": []
    }
    for res in response.results:
        ground_truth = res.user_class
        prediction = res.target_prediction
        is_error = ground_truth != prediction
        results["prediction"].append(prediction)
        results["ground_truth"].append(ground_truth)
        results["is_error"].append(is_error)
        results["class_frequency"].append(round(res.user_class_frequency * 100, 2))
        distances = []
        for explanation in res.explanations:
            distances.append(explanation.distance)
        results["min_distance"].append(min(distances))
        results["max_distance"].append(max(distances))
        results["avg_distance"].append(sum(distances) / len(distances))
        if is_error:
            pass
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["is_error", "class_frequency"], ascending=False).reset_index(drop=True)
    return results_df
