import os
import re

import boto3
from dotenv import load_dotenv

# from neo4j import GraphDatabase
from openai import AzureOpenAI

from common.logs import get_logger

# Load environment variables
load_dotenv()

# Initialize the logger
logger = get_logger()

# Cached resources
_neo4j_driver = None
_embedding_client = None
_embedding_model_name = None
_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        try:
            region_name = os.getenv("REGION_NAME")
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            endpoint_url = os.getenv("AWS_ENDPOINT_URL")

            # If running in Docker, replace "localhost" with "host.docker.internal"
            # if endpoint_url and "localhost" in endpoint_url:
            #     endpoint_url = endpoint_url.replace("localhost", "host.docker.internal")

            _s3_client = boto3.client(
                service_name="s3",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
            )
            logger.info("S3 client successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise
    return _s3_client


# def get_neo4j_driver():
#     """Lazy initialization for Neo4j driver."""
#     global _neo4j_driver
#     if _neo4j_driver is None:
#         try:
#             neo4j_url = os.getenv("NEO4J_URL")
#             neo4j_user = os.getenv("NEO4J_USER")
#             neo4j_password = os.getenv("NEO4J_PASSWORD")
#             if not all([neo4j_url, neo4j_user, neo4j_password]):
#                 raise ValueError("Missing one or more Neo4j environment variables.")
#             _neo4j_driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
#             logger.info(f"Neo4j connection initialized at {neo4j_url}")
#         except Exception as e:
#             logger.error(f"Failed to initialize Neo4j driver: {e}")
#             raise
#     return _neo4j_driver


def get_embedding_client():
    """Lazy initialization for OpenAI embedding client."""
    global _embedding_client
    global _embedding_model_name
    if _embedding_client is None:
        try:
            embedding_model_endpoint = os.getenv("EMBEDDING_MODEL_ENDPOINT")
            embedding_model_api_key = os.getenv("EMBEDDING_MODEL_API_KEY")
            if not all([embedding_model_endpoint, embedding_model_api_key]):
                raise ValueError("Missing one or more embedding model environment variables.")

            # Extract Azure OpenAI parameters from the endpoint
            base_url_pattern = r"https://[^/]+"
            azure_endpoint = re.search(base_url_pattern, embedding_model_endpoint).group(0)

            api_version_pattern = r"api-version=([\d-]+)"
            api_version = re.search(api_version_pattern, embedding_model_endpoint).group(1)

            deployment_pattern = r"/deployments/([^/]+)/embeddings"
            _embedding_model_name = re.search(deployment_pattern, embedding_model_endpoint).group(1)

            _embedding_client = AzureOpenAI(
                api_key=embedding_model_api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_deployment=_embedding_model_name,
            )
            logger.info("Embedding model client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            raise
    return _embedding_client, _embedding_model_name


def close_resources():
    """Close all initialized resources gracefully."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        logger.info("Neo4j driver connection closed.")
        _neo4j_driver = None
