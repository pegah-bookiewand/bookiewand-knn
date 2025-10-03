from typing import Dict, Any

import pandas as pd

from ai.utils.initialization import get_logger
from ai.utils.data_types import (
    ModelHyperparameters,
    JournalLinePredictionResponse
)
from ai.ml.knn.inference_preprocessing import preprocessing_step
from ai.ml.knn.inference_postprocessing import postprocessing_step


logger = get_logger()


def knn_inference(
        tenant_id: str,
        table_name: str,
        df: pd.DataFrame,
        model_data: Dict[str, Any],
) -> JournalLinePredictionResponse:
    """
    Perform KNN inference using the provided model data.

    Parameters
    ----------
    tenant_id : str
        The tenant identifier.
    table_name: str
    df : pd.DataFrame
        The input data for inference.
    model_data : Dict[str, Any]
        Dictionary containing trained model and preprocessing artifacts.
    
     generate_missing_embeddings : bool, optional
        Whether to generate fresh embeddings for text values not found in the pre-generated embeddings.
        Default is True, which provides better accuracy but requires calls to the embedding API.
        Set to False for faster inference without API calls, but potentially less accurate for new descriptions.

    Returns
    -------
    BatchGSTValidationResponse
        The post-processed inference response.
    """
    logger.info(f"Running KNN inference for table {table_name} tenant: {tenant_id}")
    logger.debug(f"Input dataframe shape: {df.shape}")

    # ================== Model Data ==================
    # hyperparameters
    hyperparameters = model_data["hyperparameters"]
    categorical_features = hyperparameters["categorical_features"]
    numerical_features = hyperparameters["numerical_features"]
    target_variable = hyperparameters["target_variable"]
    model_hyperparameters = ModelHyperparameters(**hyperparameters)

    # encoders
    nan_imputer_value = model_data["encoders"]["nan_imputer_value"]
    numerical_scaler = model_data["encoders"]["numerical_scaler"]
    categorical_encoder = model_data["encoders"]["categorical_encoder"]

    # model
    knn_model = model_data["knn_model"]

    # data
    train_df = model_data["train_df"]
    raw_df = model_data["raw_df"]

    # ================== Preprocessing ==================
    X = preprocessing_step(df=df,
                           categorical_features=categorical_features,
                           numerical_features=numerical_features,
                           target_variable=target_variable,
                           nan_imputer_value=nan_imputer_value,
                           numerical_scaler=numerical_scaler,
                           categorical_encoder=categorical_encoder,
                           logger=logger)

    # ================== Inference ==================
    try:
        predictions = knn_model.predict(X)
        logger.info(f"Predictions made for {len(predictions)} samples.")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise Exception

    # post-processing the predictions
    response = postprocessing_step(tenant_id=tenant_id,
                                   table_name=table_name,
                                   predictions=predictions,
                                   X=X,
                                   df=df,
                                   train_df=train_df,
                                   raw_df=raw_df,
                                   train_class_counts=hyperparameters["class_counts"],
                                   train_removed_classes=hyperparameters["removed_classes"],
                                   knn_model=knn_model,
                                   model_hyperparameters=model_hyperparameters,
                                   logger=logger)
    logger.info("Inference completed successfully.")
    return response
