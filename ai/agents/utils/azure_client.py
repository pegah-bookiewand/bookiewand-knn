import re
from urllib.parse import urlparse, parse_qs
from typing import Dict
import os
from openai import AsyncAzureOpenAI, AsyncOpenAI

import logging

logger = logging.getLogger(__name__)


def extract_azure_chat_client_info(endpoint_url: str) -> Dict[str, str]:
    # Parse the URL
    logger.info(f"Endpoint URL: {endpoint_url}")
    parsed_url = urlparse(endpoint_url)

    # Get the base URL (azure_endpoint)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    azure_endpoint = f"{scheme}://{netloc}"

    # Extract api_version from query parameters
    query_params = parse_qs(parsed_url.query)
    api_version = query_params.get('api-version', [''])[0]

    # Extract model_name from the path
    path = parsed_url.path
    # Use regex to extract the deployment name (model_name)
    deployment_match = re.search(r'/deployments/([^/]+)', path)
    model_name = deployment_match.group(1) if deployment_match else "gpt-4.1"
    
    client_info = {
        "azure_endpoint": azure_endpoint,
        "api_version": api_version,
        "model_name": model_name
    }
    return client_info


def get_azure_openai_client(
        azure_endpoint: str,
        model_name: str,
        api_version: str,
        api_key: str
) -> AsyncAzureOpenAI:
    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=model_name
    )
    return client

def get_openai_client(api_key: str) -> AsyncOpenAI:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')
    )
    return client
