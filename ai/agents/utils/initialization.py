import logging
import os
from typing import Tuple, Dict

from openai import AsyncAzureOpenAI, AsyncOpenAI
from agents.model_settings import ModelSettings
from agents import (
    set_default_openai_client,
    set_tracing_disabled
)

from ai.agents.utils.azure_client import (
    extract_azure_chat_client_info,
    get_azure_openai_client,
    get_openai_client
)


def get_llm(
        logger: logging.Logger
) -> Tuple[AsyncOpenAI, ModelSettings, Dict[str, str]]:
    """
    Initialize and configure OpenAI client for LLM inference.

    Sets up OpenAI client with deterministic model settings and extracts
    client configuration from environment variables. Configures default client
    and disables tracing for the session. Supports both Azure OpenAI and OpenRouter.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for recording setup information and configuration details.

    Returns
    -------
    Tuple[AsyncOpenAI, ModelSettings, Dict[str, str]]
        A tuple containing:
        - AsyncOpenAI: Configured OpenAI client instance (Azure or OpenRouter)
        - ModelSettings: Deterministic model settings with temperature=0.0
        - Dict[str, str]: Client configuration information including model name,
          API version, and endpoint
    """
    # Determine which provider to use based on environment variable
    llm_provider = os.getenv('LLM_PROVIDER', 'azure').lower()
    
    LLM_ENDPOINT = os.getenv('CHAT_MODEL_ENDPOINT')
    if isinstance(LLM_ENDPOINT, bytes):
        LLM_ENDPOINT = LLM_ENDPOINT.decode('utf-8')
    LLM_API_KEY = os.getenv('CHAT_MODEL_API_KEY')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if llm_provider == 'azure':
        # Use Azure OpenAI provider
        logger.info("Setting up Azure OpenAI client")
        chat_client_info = extract_azure_chat_client_info(endpoint_url=LLM_ENDPOINT)
        logger.info(f"Using Azure model: {chat_client_info['model_name']}, API version: {chat_client_info['api_version']}")
        openai_client = get_azure_openai_client(
            azure_endpoint=chat_client_info["azure_endpoint"],
            model_name=chat_client_info["model_name"],
            api_version=chat_client_info["api_version"],
            api_key=LLM_API_KEY
        )
    else:
        # Default to OpenRouter provider
        logger.info("Setting up OpenRouter client")
        chat_client_info = {
            "model_name": os.getenv('OPENROUTER_MODEL', 'openai/gpt-4.1-nano'),
            "api_version": "2025-01-01-preview",
            "azure_endpoint": ""
        }
        logger.info(f"Using OpenRouter model: {chat_client_info['model_name']}")
        openai_client = get_openai_client(api_key=openrouter_api_key)

    deterministic_settings = ModelSettings(
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=None,
        truncation="auto"
    )
    logger.info("Model settings configured for deterministic results")
    logger.info(deterministic_settings.to_json_dict())
    set_default_openai_client(client=openai_client)
    set_tracing_disabled(disabled=True)
    return openai_client, deterministic_settings, chat_client_info
