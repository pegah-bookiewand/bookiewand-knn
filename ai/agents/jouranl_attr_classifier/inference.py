from typing import Tuple, Dict, Any
import logging

import numpy as np
import pandas as pd
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel
)
from openai import AsyncAzureOpenAI
from agents.model_settings import ModelSettings
from sklearn.metrics.pairwise import cosine_similarity

from ai.agents.jouranl_attr_classifier.data_types.general import (
    LLMJournalAttributeInferenceOutput,
    LLMJournalLinePredictionSimilarExamples,
    LLMJournalLinePredictionResult,
    LLMJournalLinePredictionResponse,
    Hyperparameters,
)
from ai.agents.jouranl_attr_classifier.prompts.account_mapping import (
    generate_account_mapping_system_prompt,
    generate_account_mapping_user_prompt
)
from ai.agents.jouranl_attr_classifier.prompts.gst_validation import (
    generate_gst_validation_system_prompt,
    generate_gst_validation_user_prompt
)
from ai.utils.data_types import JournalLineRecord
from ai.ml.knn.inference_postprocessing import convert_float_string_to_int_string


def get_temporal_neighbors(
        df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str,
        k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest temporal neighbors for each test record based on date proximity.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame containing historical records with date information.
    test_df : pd.DataFrame
        Test DataFrame containing records for which to find temporal neighbors.
    date_column : str
        Name of the column containing date information in both DataFrames.
    k : int
        Number of nearest temporal neighbors to find for each test record.

    Returns
    -------
    neighbor_indices : np.ndarray
        Array of shape (n_test, k) containing indices of k nearest temporal neighbors
        for each test record.
    neighbor_distances : np.ndarray
        Array of shape (n_test, k) containing temporal distances in days between
        each test record and its k nearest neighbors.
    """
    # Convert date columns to datetime
    df_dates = pd.to_datetime(df[date_column])
    test_dates = pd.to_datetime(test_df[date_column])

    # Initialize result arrays
    n_test = len(test_df)
    n_train = len(df)
    k = min(k, n_train)  # Ensure k doesn't exceed training data size

    neighbor_indices = np.zeros((n_test, k), dtype=int)
    neighbor_distances = np.zeros((n_test, k), dtype=float)

    # For each test sample, find k closest temporal neighbors
    for i in range(n_test):
        test_date = test_dates.iloc[i]

        # Calculate temporal distances (in days) from test date to all training dates
        temporal_distances = np.abs((df_dates - test_date).dt.days)

        # Get indices of k smallest distances
        closest_indices = np.argsort(temporal_distances)[:k]
        closest_distances = temporal_distances.iloc[closest_indices].values

        neighbor_indices[i] = closest_indices
        neighbor_distances[i] = closest_distances

    return neighbor_indices, neighbor_distances


def postprocessing(
        tenant_id: str,
        table_name: str,
        id_column: str,
        target_variable: str,
        test_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        df: pd.DataFrame,
        neighbor_indices: Dict[str, np.ndarray],
        neighbor_distances: Dict[str, np.ndarray],
        inference_outputs: Dict[int, Dict[str, Any]],
        context: Dict[str, Any],
        chat_client_info: Dict[str, str],
        model_settings: ModelSettings
) -> LLMJournalLinePredictionResponse:
    """
    Post-process LLM inference results to create structured prediction responses.

    This function processes the raw inference outputs from the LLM to create a structured
    response containing predictions, confidence scores, reasoning, and similar examples
    for each test record. It consolidates the results with hyperparameters and similar
    examples based on feature similarity and distance calculations.

    Parameters
    ----------
    tenant_id : str
        The tenant identifier for the prediction request.
    table_name : str
        The name of the table being processed for account mapping.
    id_column : str
        The name of the ID column in the dataframe.
    target_variable : str
        The name of the target variable being predicted (e.g., 'account_code').
    test_df : pd.DataFrame
        The DataFrame containing test records for which predictions were made.
    raw_df : pd.DataFrame
        The raw DataFrame containing reference records.
    df : pd.DataFrame
        The training DataFrame containing reference records for similarity calculations.
    neighbor_indices : Dict[str, np.ndarray]
        Dictionary mapping feature names to arrays of neighbor indices for each test record.
    neighbor_distances : Dict[str, np.ndarray]
        Dictionary mapping feature names to arrays of neighbor distances for each test record.
    inference_outputs : Dict[int, Dict[str, Any]]
        Dictionary mapping test record indices to their inference results from the LLM.
    context : Dict[str, Any]
        Dictionary containing hyperparameters and other context information from training.
    chat_client_info : Dict[str, str]
        Dictionary containing chat client configuration including model name and API settings.
    model_settings : ModelSettings
        Configuration settings for the LLM model including temperature and other parameters.

    Returns
    -------
    LLMJournalLinePredictionResponse
        Structured response containing tenant_id, prediction results with confidence scores
        and reasoning for each test record, explanations with similar examples, and
        complete hyperparameters used for the inference.
    """

    results = []
    for test_idx in range(len(test_df)):
        if test_idx not in inference_outputs:
            logging.warning(f"No inference output for test_idx {test_idx}, skipping.")
            continue

        # query record
        test_record = test_df.iloc[[test_idx]].to_dict('records')[0]
        test_record[target_variable] = convert_float_string_to_int_string(test_record[target_variable])
        # query user_class
        test_user_class = test_record[target_variable]
        test_record = JournalLineRecord(**{**test_record, "table": table_name})

        # query target_prediction
        test_inference_output = inference_outputs[test_idx]["inference"]
        if target_variable == "account_code":
            predicted_codes = df[df['account_name'] == test_inference_output.target_variable][target_variable].unique()
            if len(predicted_codes) > 0:
                target_prediction = predicted_codes[0]
            else:
                logging.warning(f"Could not find account_code for predicted account_name: {test_inference_output.target_variable}. Skipping record.")
                continue
        elif target_variable == "tax_type":
            predicted_tax_types = df[df['tax_type'] == test_inference_output.target_variable][target_variable].unique()
            if len(predicted_tax_types) > 0:
                target_prediction = predicted_tax_types[0]
            else:
                logging.warning(f"Could not find tax_type for predicted tax_type: {test_inference_output.target_variable}. Skipping record.")
                continue
        else:
            raise NotImplementedError
        target_prediction = convert_float_string_to_int_string(target_prediction)

        # query confidence score and reasoning
        confidence_score = test_inference_output.confidence_score
        reasoning = test_inference_output.reasoning

        # similar examples
        explanation_examples = []
        for feature in neighbor_indices.keys():
            similar_indices = neighbor_indices[feature][test_idx]
            similar_distances = neighbor_distances[feature][test_idx]

            for idx, explanation_index in enumerate(similar_indices):
                explanation_distance = similar_distances[idx]
                explanation_record_id = df.iloc[explanation_index][id_column]
                explanation_record = raw_df[
                    raw_df[id_column] == explanation_record_id
                ].to_dict('records')[0]
                explanation_record[target_variable] = convert_float_string_to_int_string(
                    explanation_record[target_variable]
                )
                explanation_record = JournalLineRecord(**{**explanation_record, "table": table_name})
                
                explanation_example = LLMJournalLinePredictionSimilarExamples(
                    feature=feature,
                    train_index=explanation_index,
                    distance=explanation_distance,
                    record=explanation_record
                )
                explanation_examples.append(explanation_example)

        test_result = LLMJournalLinePredictionResult(record=test_record,
                                                     target_prediction=target_prediction,
                                                     user_class=test_user_class or "",
                                                     confidence_score=confidence_score,
                                                     reasoning=reasoning,
                                                     explanations=explanation_examples)
        results.append(test_result)

    hyperparameters = Hyperparameters(run_timestamp=context["hyperparameters"]["run_timestamp"],
                                      table_name=context["hyperparameters"]["table_name"],
                                      vectorizer=context["hyperparameters"]["model_name"],
                                      raw_data_path=context["hyperparameters"]["raw_data_path"],
                                      checkpoints_dir=context["hyperparameters"]["knn_checkpoints_dir"],
                                      features=context["hyperparameters"]["features"],
                                      date_column=context["hyperparameters"]["date_column"],
                                      target_variable=context["hyperparameters"]["target_variable"],
                                      n_neighbors=context["hyperparameters"]["n_neighbors"],
                                      num_samples=context["hyperparameters"]["num_samples"],
                                      llm_azure_endpoint=chat_client_info["azure_endpoint"],
                                      llm_api_version=chat_client_info["api_version"],
                                      llm_model_name=chat_client_info["model_name"],
                                      llm_temperature=model_settings.temperature,
                                      llm_top_p=model_settings.temperature)
    # Ensure tenant_id is a valid string, use empty string as fallback
    valid_tenant_id = tenant_id if tenant_id is not None else ""
    
    inference_response = LLMJournalLinePredictionResponse(tenant_id=valid_tenant_id,
                                                          results=results,
                                                          hyperparameters=hyperparameters)
    return inference_response


import asyncio

async def _run_inference_for_record(
        input_data_index: int,
        input_data: pd.DataFrame,
        df: pd.DataFrame,
        features: list,
        neighbor_indices: dict,
        agent: Agent,
        target_variable: str,
        logger: logging.Logger,
        semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    async with semaphore:
        try:
            logger.info(f"Processing test record {input_data_index + 1}/{len(input_data)}")

            query_df = input_data.iloc[[input_data_index]][features]
            similar_records_by_feature = {}
            for col in neighbor_indices.keys():
                col_neighbor_indices = neighbor_indices[col][input_data_index]
                if len(col_neighbor_indices) > 0:
                    col_neighbors = df.iloc[col_neighbor_indices].reset_index(drop=True)
                    similar_records_by_feature[col] = col_neighbors
                else:
                    similar_records_by_feature[col] = pd.DataFrame()
                    logger.info(f"Warning: No neighbors found for feature {col} in test record {input_data_index}")

            if target_variable == "account_code":
                query_user_prompt = generate_account_mapping_user_prompt(
                    query_df=query_df,
                    similar_records_by_feature=similar_records_by_feature
                )
            elif target_variable == "tax_type":
                query_user_prompt = generate_gst_validation_user_prompt(
                    query_df=query_df,
                    similar_records_by_feature=similar_records_by_feature
                )
            else:
                raise ValueError(f"Unsupported target variable: {target_variable}. "
                                 f"Target variables are in (account_code, tax_type)")
            logger.info(f"Record {input_data_index} - Generated user prompt with {len(query_user_prompt)} characters")

            result = await Runner.run(starting_agent=agent, input=query_user_prompt)
            output = result.final_output_as(LLMJournalAttributeInferenceOutput)
            logger.info(f"Record {input_data_index} - Inference successful: "
                        f"predicted_target_variable='{output.target_variable}', "
                        f"confidence={output.confidence_score:.3f}")
            return {
                "test_idx": input_data_index,
                "record": query_df.to_dict('records')[0],
                "inference": output
            }
        except Exception as e:
            logger.error(f"Record {input_data_index} - Inference failed: {str(e)}")
            return None

async def llm_inference(
        tenant_id: str,
        table_name: str,
        input_data: pd.DataFrame,
        context: Dict[str, Any],
        client: AsyncAzureOpenAI,
        chat_client_info: Dict[str, str],
        model_settings: ModelSettings,
        logger: logging.Logger,
        include_date_column: bool = False,
        n_workers: int = 8,
) -> LLMJournalLinePredictionResponse:
    """
    Perform LLM-based inference for GST validation/account mapping prediction using nearest neighbors.

    This function uses a large language model to predict GST validation/account mappings for journal
    line records by finding similar records using TF-IDF vectorization and temporal proximity,
    then prompting the LLM to make predictions based on these similar examples.

    Parameters
    ----------
    tenant_id : str
        The tenant identifier for the prediction request.
    table_name : str
        The name of the table being processed for GST validation/account mapping.
    test_df : pd.DataFrame
        The DataFrame containing test records for which to predict GST validation/account mappings.
    context : Dict[str, Any]
        Dictionary containing trained model context including dataframe, hyperparameters,
        feature tokenizers, and token matrices for similarity calculations.
    client : AsyncAzureOpenAI
        The Azure OpenAI client instance for making LLM inference calls.
    chat_client_info : Dict[str, str]
        Dictionary containing chat client configuration including model name and API settings.
    model_settings : ModelSettings
        Configuration settings for the LLM model including temperature and other parameters.
    logger : logging.Logger
        Logger instance for recording inference progress and debugging information.
    include_date_column : bool, optional (default=False)
        Boolean indicating whether to include the date column in the feature list

    Returns
    -------
    LLMJournalLinePredictionResponse
        Response object containing tenant_id, prediction results with confidence scores
        and reasoning for each test record, and model hyperparameters.
    """
    # Validate tenant_id at the start of the function
    if tenant_id is None:
        logger.error("tenant_id is None at the start of llm_inference")
        # Create a valid, empty response to prevent downstream errors
        empty_hyperparameters = Hyperparameters(
            run_timestamp="",
            table_name=table_name,
            vectorizer="",
            raw_data_path="",
            checkpoints_dir="",
            features=[],
            date_column="",
            target_variable="",
            n_neighbors=0,
            num_samples=0,
            llm_azure_endpoint="",
            llm_api_version="",
            llm_model_name="",
            llm_temperature=0.0,
            llm_top_p=0.0,
        )
        return LLMJournalLinePredictionResponse(
            tenant_id="",
            results=[],
            hyperparameters=empty_hyperparameters
        )
    
    try:
        logger.info(f"Starting LLM inference for tenant_id: {tenant_id}, table: {table_name}")
        logger.info(f"Input data shape: {input_data.shape}")

        df = context["df"]
        raw_df = context["raw_df"]
        id_column = context["hyperparameters"]["id_column"]
        features = context["hyperparameters"]["features"]
        constraint_features = context["hyperparameters"]["constraint_features"]
        date_column = context["hyperparameters"]["date_column"]
        target_variable = context["hyperparameters"]["target_variable"]
        n_neighbors = context["hyperparameters"]["n_neighbors"]

        logger.info(f"Training data shape: {df.shape}")
        logger.info(f"Features: {features}")
        logger.info(f"Date column: {date_column}")
        logger.info(f"Target variable: {target_variable}")
        logger.info(f"Number of neighbors: {n_neighbors}")

        if target_variable == "account_code":
            unique_account_names = [x for x in df['account_name'].unique().tolist() if pd.notna(x)]
            logger.info(f"Found {len(unique_account_names)} unique account names")
            system_prompt = generate_account_mapping_system_prompt(feature_list=features,
                                                                   unique_account_names=unique_account_names)
        elif target_variable == "tax_type":
            unique_tax_types = [x for x in df['tax_type'].unique().tolist() if pd.notna(x)]
            logger.info(f"Found {len(unique_tax_types)} unique tax types")
            system_prompt = generate_gst_validation_system_prompt(feature_list=features,
                                                                  unique_tax_types=unique_tax_types)
        else:
            raise ValueError(f"Unsupported target variable: {target_variable}. "
                             f"Target variables are in (account_code, tax_type)")
        logger.info(f"Generated system prompt with {len(system_prompt)} characters")

        agent = Agent(
            name=f"{target_variable}_classifier",
            instructions=system_prompt,
            output_type=LLMJournalAttributeInferenceOutput,
            model=OpenAIChatCompletionsModel(
                model=chat_client_info["model_name"],
                openai_client=client,
            ),
            model_settings=model_settings
        )
        logger.info(f"Initialized agent for {target_variable} classification "
                    f"using model: {chat_client_info['model_name']}")

        # extracting feature neighbors
        logger.info("Starting feature neighbor extraction")
        neighbor_indices = {}
        neighbor_distances = {}
        for feat in features:
            logger.info(f"Processing feature: {feat}")

            feat_tokenizer = context["feature_tokenizer"][feat]
            feat_token_matrix = context["feature_token_matrix"][feat]

            test_feature_data = input_data[feat].fillna('').astype(str)
            test_tfidf_matrix = feat_tokenizer.transform(test_feature_data)
            logger.info(f"Feature {feat} - Test TF-IDF matrix shape: {test_tfidf_matrix.shape}")

            # Initialize lists to store variable-length neighbors for each test record
            feat_top_indices_list = []
            feat_top_similarities_list = []

            # Process each test record individually to apply constraints
            for input_data_index in range(len(input_data)):
                # Get constraint values for current test record
                test_constraints = input_data.iloc[input_data_index][constraint_features]

                # Find training records that match ALL constraint features
                constraint_mask = pd.Series([True] * len(df))
                for constraint_feat in constraint_features:
                    constraint_mask &= (df[constraint_feat] == test_constraints[constraint_feat])

                # Excluding the data point itself from similarity computation
                test_record_line_item_id = input_data.iloc[input_data_index]['line_item_id']
                self_test_record = raw_df[raw_df['line_item_id'] == test_record_line_item_id]
                if len(self_test_record) == 1:
                    self_index = self_test_record.index.item()
                    constraint_mask[self_index] = False

                # Get indices of valid training records
                valid_indices = np.where(constraint_mask)[0]

                if len(valid_indices) == 0:
                    logger.warning(f"No training records found matching constraints for test record {input_data_index}")
                    # Store empty arrays for this test record
                    feat_top_indices_list.append(np.array([], dtype=int))
                    feat_top_similarities_list.append(np.array([]))
                    continue

                # Calculate similarity only for valid training records
                test_record_tfidf = test_tfidf_matrix[input_data_index:input_data_index + 1]  # Keep as 2D matrix
                valid_feat_token_matrix = feat_token_matrix[valid_indices]

                record_similarity = cosine_similarity(test_record_tfidf, valid_feat_token_matrix).flatten()

                # Get top n_neighbors (or all if fewer available)
                k = min(n_neighbors, len(valid_indices))
                top_k_positions = np.argsort(record_similarity)[-k:][::-1]  # Get indices in descending order

                # Map back to original dataframe indices
                selected_indices = valid_indices[top_k_positions]
                selected_similarities = record_similarity[top_k_positions]

                # Store results
                feat_top_indices_list.append(selected_indices)
                feat_top_similarities_list.append(selected_similarities)

                if k < n_neighbors:
                    logger.warning(f"Warning: Only found {k} valid neighbors for test record {input_data_index}, "
                                f"requested {n_neighbors}")
            # Convert to arrays, but keep the variable length structure
            neighbor_indices[feat] = feat_top_indices_list

            # Calculate distances
            feat_top_distances_list = [1 - similarities for similarities in feat_top_similarities_list]
            neighbor_distances[feat] = feat_top_distances_list
            # Log similarity statistics
            all_similarities = np.concatenate([sim for sim in feat_top_similarities_list if len(sim) > 0])
            if len(all_similarities) > 0:
                logger.info(f"Feature {feat} - Found constrained neighbors with similarity range: "
                            f"[{all_similarities.min():.4f}, {all_similarities.max():.4f}]")
            else:
                logger.info(f"Warning: Feature {feat} - No valid neighbors found for any test records")

        # extracting temporal neighbors
        if include_date_column:
            logger.info("Starting temporal neighbor extraction")
            temporal_neighbor_indices, temporal_neighbor_distances = get_temporal_neighbors(df=df,
                                                                                            test_df=input_data,
                                                                                            date_column=date_column,
                                                                                            k=n_neighbors)
            neighbor_indices[date_column] = temporal_neighbor_indices
            neighbor_distances[date_column] = temporal_neighbor_distances
            logger.info(f"Temporal neighbors - Distance range: "
                        f"[{temporal_neighbor_distances.min():.2f}, "
                        f"{temporal_neighbor_distances.max():.2f}] days")

        # inference
        logger.info(f"Starting LLM inference for individual records with {n_workers} workers")
        semaphore = asyncio.Semaphore(n_workers)
        tasks = [
            _run_inference_for_record(
                input_data_index=input_data_index,
                input_data=input_data,
                df=df,
                features=features,
                neighbor_indices=neighbor_indices,
                agent=agent,
                target_variable=target_variable,
                logger=logger,
                semaphore=semaphore
            )
            for input_data_index in range(len(input_data))
        ]
        
        results = await asyncio.gather(*tasks)
        
        inference_outputs = {res["test_idx"]: res for res in results if res}
        logger.info(f"Completed LLM inference for all {len(input_data)} records")

        # Postprocessing
        logger.info("Starting postprocessing")

        inference_response = postprocessing(
            tenant_id=tenant_id,
            table_name=table_name,
            id_column=id_column,
            target_variable=target_variable,
            test_df=input_data,
            raw_df=raw_df,
            df=df,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            inference_outputs=inference_outputs,
            context=context,
            chat_client_info=chat_client_info,
            model_settings=model_settings
        )
        logger.info(f"LLM inference completed successfully for tenant_id: {tenant_id}, table: {table_name}")
        logger.info(f"Total predictions generated: {len(inference_response.results)}")
        logger.info(f"Inference response hyperparameters: {inference_response.hyperparameters}")
        return inference_response
    except Exception as e:
        logger.error(f"An unexpected error occurred during llm_inference for tenant {tenant_id}, table {table_name}: {e}", exc_info=True)
        # Create a valid, empty response to prevent downstream errors
        empty_hyperparameters = Hyperparameters(
            run_timestamp="",
            table_name=table_name,
            vectorizer="",
            raw_data_path="",
            checkpoints_dir="",
            features=[],
            date_column="",
            target_variable="",
            n_neighbors=0,
            num_samples=0,
            llm_azure_endpoint="",
            llm_api_version="",
            llm_model_name="",
            llm_temperature=0.0,
            llm_top_p=0.0,
        )
        # Ensure tenant_id is a valid string, use empty string as fallback
        valid_tenant_id = tenant_id if tenant_id is not None else ""
        
        return LLMJournalLinePredictionResponse(
            tenant_id=valid_tenant_id,
            results=[],
            hyperparameters=empty_hyperparameters
        )

