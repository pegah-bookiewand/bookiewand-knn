import os
import boto3
import logging
from typing import List, Set
from common.logs import get_logger

logger = get_logger()


def create_bucket(
        s3_client: boto3.client,
        bucket_name: str
) -> None:
    """
    Creates an S3 bucket if it does not already exist.

    Parameters
    ----------
    s3_client : boto3.client
        The Boto3 S3 client.
    bucket_name : str
        The name of the S3 bucket to create.

    Returns
    -------
    None
    """
    # Get the list of existing buckets
    existing_buckets = [bucket['Name'] for bucket in s3_client.list_buckets().get('Buckets', [])]

    if bucket_name in existing_buckets:
        logger.info(f"Bucket '{bucket_name}' already exists. No action taken.")
    else:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' created successfully.")


def upload_directory_to_s3(
        s3_client: boto3.client,
        local_directory: str,
        bucket_name: str,
        prefix: str
) -> None:
    """
    Uploads all files from a local directory to an S3 bucket under the specified prefix.

    Parameters
    ----------
    s3_client : boto3.client
        The Boto3 S3 client.
    local_directory : str
        The local directory containing files to upload.
    bucket_name : str
        The S3 bucket name.
    prefix : str
        The S3 prefix (folder) under which files will be uploaded.

    Returns
    -------
    None
    """
    if not os.path.isdir(local_directory):
        logger.error(f"Error: '{local_directory}' is not a valid directory.")
        return

    for root, subdirectories, files in os.walk(local_directory):
        for filename in files:
            # full path to the local file
            local_file_path = os.path.join(root, filename)
            # compute the relative path from local_directory and ensure '/' separators
            relative_path = os.path.relpath(local_file_path, local_directory).replace(os.sep, '/')
            # construct the S3 key, handling empty prefix case
            if prefix:
                s3_key = f"{prefix.rstrip('/')}/{relative_path}"
            else:
                s3_key = relative_path

            # upload the file to S3
            s3_client.upload_file(local_file_path, bucket_name, s3_key)
            logger.info(f"Uploaded '{local_file_path}' to 's3://{bucket_name}/{s3_key}'")


def upload_file_to_s3(
        s3_client: boto3.client,
        file_path: str,
        bucket_name: str,
        prefix: str
) -> None:
    """
    Uploads a single file to an S3 bucket under the specified prefix.

    Parameters
    ----------
    s3_client : boto3.client
        The Boto3 S3 client.
    file_path : str
        The local file path to upload.
    bucket_name : str
        The S3 bucket name.
    prefix : str
        The S3 prefix (folder) under which the file will be uploaded.

    Returns
    -------
    None
    """
    if not os.path.isfile(file_path):
        logger.error(f"Error: '{file_path}' is not a valid file.")
        return

    file_name = os.path.basename(file_path)
    s3_key = f"{prefix}/{file_name}" if prefix else file_name

    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logger.info(f"Uploaded '{file_path}' to 's3://{bucket_name}/{s3_key}'")
    except Exception as e:
        logger.error(f"Failed to upload '{file_path}' to 's3://{bucket_name}/{s3_key}': {e}")


def download_s3_directory(
        s3_client: boto3.client,
        bucket_name: str,
        prefix: str,
        local_data_dir: str
) -> None:
    """
    Download all objects from an S3 bucket under a specified prefix to a local directory.

    Parameters
    ----------
    s3_client : boto3.client
        The S3 client used to interact with AWS S3.
    bucket_name : str
        The name of the S3 bucket.
    prefix : str
        The prefix (folder path) within the S3 bucket to download files from.
    local_data_dir : str
        The local directory where files will be saved.
    logger: logging.Logger

    Returns
    -------
    None
        This function does not return anything but downloads the files to the specified local directory.
    """
    logger.info("Starting the download of files from S3 bucket: %s, prefix: %s", bucket_name, prefix)

    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir, exist_ok=True)
        logger.info("Created local directory: %s", local_data_dir)

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                file_name = os.path.relpath(s3_key, prefix)  # Remove prefix from the key
                local_file_path = os.path.join(local_data_dir, file_name)

                if not os.path.exists(os.path.dirname(local_file_path)):
                    os.makedirs(os.path.dirname(local_file_path))

                logger.info("Downloading %s to %s", s3_key, local_file_path)
                s3_client.download_file(bucket_name, s3_key, local_file_path)
                logger.info("Successfully downloaded %s", s3_key)

    logger.info("Finished downloading all files from S3.")


def list_s3_directories(
        s3_client: boto3.client,
        bucket_name: str,
        prefix: str
) -> List[str]:
    """
    List only the first-level directories under a specified prefix in an S3 bucket.

    Parameters
    ----------
    s3_client : boto3.client
        The S3 client used to interact with AWS S3.
    bucket_name : str
        The name of the S3 bucket.
    prefix : str
        The prefix (folder path) within the S3 bucket to list directories from.

    Returns
    -------
    List[str] or None
        A list of first-level directory paths under the given prefix.
        Returns None if the bucket does not exist.
    """
    logger.info("Listing first-level directories in S3 bucket: %s, prefix: %s", bucket_name, prefix)

    if not prefix.endswith("/"):
        prefix += "/"  # Ensure prefix ends with '/'

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        directories: Set[str] = set()

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    parts = s3_key[len(prefix):].split("/")  # Remove prefix and split

                    if len(parts) > 1:  # First-level directory exists
                        directories.add(parts[0])

        sorted_dirs = sorted(directories)  # Sort for consistent ordering
        logger.info("Found %d first-level directories.", len(sorted_dirs))

        return sorted_dirs
    except Exception as e:
        if "NoSuchBucket" in str(e):
            logger.warning("Bucket %s does not exist. Returning None.", bucket_name)
            return None
        else:
            # Re-raise any other exceptions
            logger.error("Error accessing S3 bucket %s: %s", bucket_name, str(e))
            raise


def empty_bucket(
        s3_client: boto3.client,
        bucket_name: str
) -> None:
    """
    Deletes all objects in an S3 bucket.

    Parameters
    ----------
    s3_client : boto3.client
        The Boto3 S3 client.
    bucket_name : str
        The S3 bucket name to be emptied.

    Returns
    -------
    None
    """
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3_client.delete_object(Bucket=bucket_name, Key=obj['Key'])
                logger.info(f"Deleted object '{obj['Key']}' from bucket '{bucket_name}'")
    except Exception as e:
        logger.error(f"Error emptying bucket {bucket_name}: {e}")


def delete_bucket(
        s3_client: boto3.client,
        buckets_to_delete: List[str]
) -> None:
    """
    Deletes multiple S3 buckets after emptying them.

    Parameters
    ----------
    s3_client : boto3.client
        The Boto3 S3 client.
    buckets_to_delete : List[str]
        A list of bucket names to delete.

    Returns
    -------
    None
    """
    for bucket_name in buckets_to_delete:
        empty_bucket(s3_client, bucket_name=bucket_name)  # Ensure bucket is empty before deletion
        s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"Deleted bucket: {bucket_name}")
