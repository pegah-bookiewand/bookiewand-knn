import os
from typing import Any, Dict

from common.logs import get_logger

from ai.utils.initialization import get_s3_client
from ai.utils.s3_utils import download_s3_directory, list_s3_directories


logger = get_logger()


def get_gst_embeddings_data() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the latest embeddings data, loading it lazily if not already available.
    This function ensures that the embeddings data is loaded before performing any inference.
    If the embeddings data is not already loaded, it attempts to load the latest version from the checkpoint directory.
    """
    s3_client = get_s3_client()
    # Download embeddings

    bucket_name = os.getenv('GST_VALIDATION_BUCKET_NAME')
    embeddings_local_base_dir = os.getenv("EMBEDDINGS_LOCAL_BASE_DIR")
    embedding_api_json_dir = os.getenv("EMBEDDING_API_JSON_DIR")
    embedding_s3_dir = os.getenv("EMBEDDINGS_DIR")


    try:
        # First list tenants using the full path structure as it appears in S3
        tenant_id_list = list_s3_directories(s3_client=s3_client,
                                         bucket_name=bucket_name,
                                         prefix=embedding_s3_dir)

        logger.info(f"Found tenant directories in S3: {tenant_id_list}")

        embeddings_data = {}
        for tenant_id in tenant_id_list:
            embeddings_data[tenant_id] = {}
            # The tenant prefix needs to include the full path as it appears in S3
            tenant_prefix = os.path.join(embedding_s3_dir, tenant_id)
            logger.debug(f"Listing embeddings for tenant {tenant_id} using prefix: {tenant_prefix}")

            # Download embeddings for this tenant
            local_embeddings_dir = os.path.join(embeddings_local_base_dir, embedding_api_json_dir, tenant_id)

            # Create directory if it doesn't exist
            os.makedirs(local_embeddings_dir, exist_ok=True)

            logger.info(f"Downloading embeddings from s3://{bucket_name}/{tenant_prefix} to {local_embeddings_dir}")

            download_s3_directory(
                s3_client=s3_client,
                bucket_name=bucket_name,
                prefix=tenant_prefix,
                local_data_dir=local_embeddings_dir
            )
        logger.info(f"Downloaded embeddings")
    except Exception as e:
        logger.warning(f"Error downloading embeddings: {str(e)}")
        logger.info(f"Continuing with model collection without embeddings")

def get_account_mapping_embeddings_data() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the latest account mapping embeddings data, loading it lazily if not already available.
    This function ensures that the embeddings data is loaded before performing any inference.
    If the embeddings data is not already loaded, it attempts to load the latest version from the checkpoint directory.
    """
    s3_client = get_s3_client()
    # Download embeddings

    bucket_name = os.getenv('ACCOUNT_MAPPING_BUCKET_NAME')
    embeddings_local_base_dir = os.getenv("ACCOUNT_MAPPING_EMBEDDINGS_LOCAL_BASE_DIR")
    embedding_api_json_dir = os.getenv("ACCOUNT_MAPPING_EMBEDDING_API_JSON_DIR")
    embedding_s3_dir = os.getenv("ACCOUNT_MAPPING_EMBEDDINGS_DIR")


    try:
        # First list tenants using the full path structure as it appears in S3
        tenant_id_list = list_s3_directories(s3_client=s3_client,
                                         bucket_name=bucket_name,
                                         prefix=embedding_s3_dir)

        logger.info(f"Found account mapping tenant directories in S3: {tenant_id_list}")

        embeddings_data = {}
        for tenant_id in tenant_id_list:
            embeddings_data[tenant_id] = {}
            # The tenant prefix needs to include the full path as it appears in S3
            tenant_prefix = os.path.join(embedding_s3_dir, tenant_id)
            logger.debug(f"Listing account mapping embeddings for tenant {tenant_id} using prefix: {tenant_prefix}")

            column_names = list_s3_directories(s3_client=s3_client, 
                                              bucket_name=bucket_name,
                                              prefix=tenant_prefix)

            logger.info(f"Found account mapping column directories for tenant {tenant_id}: {column_names}")

            for column_name in column_names:
                # Download embeddings for this tenant
                s3_tenant_embeddings_prefix = os.path.join(tenant_prefix, column_name)
                local_embeddings_dir = os.path.join(embeddings_local_base_dir, embedding_api_json_dir, tenant_id, column_name)

                # Create directory if it doesn't exist
                os.makedirs(local_embeddings_dir, exist_ok=True)

                logger.info(f"Downloading account mapping embeddings from s3://{bucket_name}/{s3_tenant_embeddings_prefix} to {local_embeddings_dir}")

                download_s3_directory(
                    s3_client=s3_client,
                    bucket_name=bucket_name,
                    prefix=s3_tenant_embeddings_prefix,
                    local_data_dir=local_embeddings_dir
                )
        logger.info(f"Downloaded account mapping embeddings")
    except Exception as e:
        logger.warning(f"Error downloading account mapping embeddings: {str(e)}")
        logger.info(f"Continuing with model collection without account mapping embeddings")