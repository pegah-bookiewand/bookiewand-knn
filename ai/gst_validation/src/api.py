import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from common.logs import get_logger
from ai.utils.data_types import (
    JournalLinePredictionRequest,
    JournalLinePredictionResponse
)
from ai.agents.jouranl_attr_classifier.data_types.general import LLMJournalLinePredictionResponse
from ai.utils.data_collection import (
    model_collection_step,
    llm_context_collection_step
)
from ai.ml.knn.inference import knn_inference
from ai.agents.utils.initialization import get_llm
from ai.agents.jouranl_attr_classifier.inference import llm_inference


load_dotenv(override=True)
logger = get_logger()


# Global variable to store the model data. It might be None if not loaded yet.
# TODO: Handle model version through a central service, in the case of multiple instances of the API
model_data: Optional[Dict[str, Dict[str, Any]]] = None
context_data: Optional[Dict[str, Dict[str, Any]]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Custom lifespan event handler for FastAPI startup and shutdown events.
    """
    # Startup event
    global model_data, context_data

    # Load model data
    try:
        bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
        gst_validation_checkpoints_prefix = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
        base_data_dir = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
        gst_validation_api_checkpoints_dir = os.getenv('GST_VALIDATION_API_CHECKPOINTS_DIR')
        model_data = model_collection_step(base_data_dir=base_data_dir,
                                           bucket_name=bucket_name,
                                           checkpoints_prefix=gst_validation_checkpoints_prefix,
                                           checkpoints_dir=gst_validation_api_checkpoints_dir,
                                           logger=logger)
        logger.info(f"Model loaded successfully on startup. "
                    f"Model Name: {model_data['name']} | "
                    f"Version: {model_data['version']}")
    except Exception as e:
        logger.info(f"Failed to load model on startup: {str(e)}")
        model_data = None

    # Load context data
    try:
        bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
        gst_validation_checkpoints_prefix = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
        base_data_dir = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
        gst_validation_api_checkpoints_dir = os.getenv('GST_VALIDATION_API_CHECKPOINTS_DIR')
        context_data = llm_context_collection_step(base_data_dir=base_data_dir,
                                                   bucket_name=bucket_name,
                                                   checkpoints_prefix=gst_validation_checkpoints_prefix,
                                                   checkpoints_dir=gst_validation_api_checkpoints_dir,
                                                   logger=logger)
        logger.info(f"LLM Context loaded successfully on startup. "
                    f"Version: {context_data['version']}")
    except Exception as e:
        logger.info(f"Failed to load LLM context on startup: {str(e)}")
        context_data = None

    yield

    logger.info("Application shutdown")


# FastAPI application instance
app = FastAPI(lifespan=lifespan)


def get_model_data() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the latest KNN model data, loading it lazily if not already available.

    This function ensures that the model data is loaded before performing any inference.
    If the model data is not already loaded, it attempts to load the latest version from the checkpoint directory.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary containing model metadata for each tenant id.

    Raises
    ------
    Exception
        If the model data fails to load due to any error.
    """
    global model_data
    if model_data is None:
        try:
            bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
            gst_validation_checkpoints_prefix = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
            base_data_dir = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
            gst_validation_api_checkpoints_dir = os.getenv('GST_VALIDATION_API_CHECKPOINTS_DIR')
            model_data = model_collection_step(base_data_dir=base_data_dir,
                                               bucket_name=bucket_name,
                                               checkpoints_prefix=gst_validation_checkpoints_prefix,
                                               checkpoints_dir=gst_validation_api_checkpoints_dir,
                                               logger=logger)
            logger.info(f"Model loaded lazily. "
                        f"Model Name: {model_data['name']} | "
                        f"Version: {model_data['version']}")
        except Exception as e:
            raise Exception(f"Failed to load model data: {e}")
    return model_data


def get_llm_context_data() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve LLM context data for gst validation inference.

    Loads and returns the LLM context data required for gst validation predictions.
    Uses a global cache to avoid reloading data on subsequent calls. The function
    retrieves configuration from environment variables and delegates the actual
    data collection to llm_context_collection_step.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing LLM context data with version information and
        tenant-specific context for gst validation inference.

    Raises
    ------
    Exception
        If loading of model data fails due to missing environment variables,
        S3 access issues, or other configuration problems.
    """
    global context_data
    if context_data is None:
        try:
            bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
            gst_validation_checkpoints_prefix = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
            base_data_dir = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
            gst_validation_api_checkpoints_dir = os.getenv('GST_VALIDATION_API_CHECKPOINTS_DIR')
            context_data = llm_context_collection_step(base_data_dir=base_data_dir,
                                                       bucket_name=bucket_name,
                                                       checkpoints_prefix=gst_validation_checkpoints_prefix,
                                                       checkpoints_dir=gst_validation_api_checkpoints_dir,
                                                       logger=logger)
            logger.info(f"LLM Context loaded. "
                        f"Version: {context_data['version']}")
        except Exception as e:
            raise Exception(f"Failed to load model data: {e}")
    return context_data


@app.post("/gst_validation", response_model=JournalLinePredictionResponse)
def gst_validation(request: JournalLinePredictionRequest) -> JournalLinePredictionResponse:
    """
    Perform GST validation for a batch of journal records using the KNN model.

    Parameters
    ----------
    request : BatchGSTValidationRequest
        The request containing a list of journal records and the tenant ID.

    Returns
    -------
    BatchGSTValidationResponse
        The response containing GST validation results, including tax name predictions
        and explanations for each record.
    """
    try:
        current_model = get_model_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model data not loaded. {str(e)}")
    try:
        if request.tenant_id not in current_model["models"]:
            available_tenants = list(current_model["models"].keys())
            raise HTTPException(
                status_code=404,
                detail=f"Tenant ID '{request.tenant_id}' not found. Available tenants: {available_tenants}"
            )

        if request.table not in current_model["models"][request.tenant_id]:
            available_tables = list(current_model["models"][request.tenant_id].keys())
            raise HTTPException(
                status_code=404,
                detail=f"Table '{request.table}' not found for tenant '{request.tenant_id}'. "
                       f"Available tables: {available_tables}"
            )

        tenant_model = current_model["models"][request.tenant_id][request.table]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error accessing tenant model: {str(e)}"
        )

    # Convert the validated data to a DataFrame
    test_df = request.to_dataframe()

    # Run inference using the provided test DataFrame and model data
    try:
        response = knn_inference(tenant_id=request.tenant_id,
                                 table_name=request.table,
                                 df=test_df,
                                 model_data=tenant_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Return the predictions as a JSON response
    return response


@app.post("/llm_gst_validation", response_model=LLMJournalLinePredictionResponse)
async def llm_gst_validation(request: JournalLinePredictionRequest) -> LLMJournalLinePredictionResponse:
    """
    HTTP endpoint for LLM-based gst validation prediction using journal line data.

    This function validates tenant and table context, converts request data to DataFrame format,
    initializes the LLM client, and performs inference to predict gst validations for journal lines.

    Parameters
    ----------
    request : JournalLinePredictionRequest
        The request object containing tenant_id, table name, and journal line data records
        to be processed for gst validation prediction.

    Returns
    -------
    LLMJournalLinePredictionResponse
        Response object containing the tenant_id, prediction results with confidence scores
        and reasoning, and model hyperparameters used for the inference.

    Raises
    ------
    HTTPException
        500 status code if LLM context data loading fails
        404 status code if tenant_id or table is not found in context
        500 status code if inference fails or unexpected errors occur during processing
    """
    try:
        current_context = get_llm_context_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM context data not loaded. {str(e)}")
    try:
        if request.tenant_id not in current_context["context"]:
            available_tenants = list(current_context["context"].keys())
            raise HTTPException(
                status_code=404,
                detail=f"Tenant ID '{request.tenant_id}' not found. Available tenants: {available_tenants}"
            )

        if request.table not in current_context["context"][request.tenant_id]:
            available_tables = list(current_context["context"][request.tenant_id].keys())
            raise HTTPException(
                status_code=404,
                detail=f"Table '{request.table}' not found for tenant '{request.tenant_id}'. "
                       f"Available tables: {available_tables}"
            )
        context = current_context["context"][request.tenant_id][request.table]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error accessing tenant model: {str(e)}"
        )

    # Run inference using the provided test DataFrame and model data
    try:
        # Convert the validated data to a DataFrame
        test_df = request.to_dataframe()

        # loading the OpenAI client
        openai_client, deterministic_settings, chat_client_info = get_llm(logger=logger)

        response = await llm_inference(
            tenant_id=request.tenant_id,
            table_name=request.table,
            input_data=test_df,
            context=context,
            client=openai_client,
            chat_client_info=chat_client_info,
            model_settings=deterministic_settings,
            logger=logger
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Return the predictions as a JSON response
    return response


@app.post("/refresh_model")
def refresh_model():
    """
    Refresh the KNN model by loading the latest checkpoint.

    Returns
    -------
    dict
        A dictionary containing the message, model name, and version after refreshing.
    """
    global model_data
    try:
        bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
        gst_validation_checkpoints_prefix = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
        base_data_dir = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
        gst_validation_api_checkpoints_dir = os.getenv('GST_VALIDATION_API_CHECKPOINTS_DIR')
        model_data = model_collection_step(base_data_dir=base_data_dir,
                                           bucket_name=bucket_name,
                                           checkpoints_prefix=gst_validation_checkpoints_prefix,
                                           checkpoints_dir=gst_validation_api_checkpoints_dir,
                                           logger=logger)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh model: {str(e)}")
    return {"message": "Model refreshed successfully", "model": model_data["name"], "version": model_data["version"]}


@app.get("/model_version")
def get_model_version():
    """
    Get the version of the currently loaded KNN model.

    Returns
    -------
    str
        The version of the currently loaded model.
    """
    try:
        current_model = get_model_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model data not loaded. {str(e)}")
    return current_model['version']
