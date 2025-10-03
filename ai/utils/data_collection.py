import os
import logging
import pickle
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

import yaml
from dotenv import load_dotenv
from sqlalchemy import text
import pandas as pd

from common.database_config import get_database_session
from ai.utils.initialization import get_s3_client
from ai.utils.s3_utils import list_s3_directories, download_s3_directory
from ai.ml.knn.preprocessing_utils import validate_model_config
from common.logs import get_logger


load_dotenv(override=True)

logger = get_logger()


def fetch_tenant_ids() -> list[str]:
    """
    Fetch all unique tenant IDs from the database.

    Parameters
    ----------
    logger : Any
        A logger instance for logging errors and execution info.

    Returns
    -------
    List[str]
        A list of unique tenant IDs.
    """
    logger.info("Fetching all unique tenant IDs from the database...")

    session = get_database_session()
    try:
        # Query to get all distinct tenant_ids from the journal table
        query = text("SELECT DISTINCT tenant_id FROM xero_tenants")
        result = session.execute(query)
        tenant_ids = [str(row[0]) for row in result]
        logger.info(f"Found {len(tenant_ids)} unique tenant IDs")
        return tenant_ids
    except Exception as e:
        session.rollback()
        logger.error(f"Error fetching tenant IDs: {e}")
        raise
    finally:
        session.close()


def fetch_database_data(
        table_query_mapping: Dict[str, str],
        logger: Any,
        tenant_id: str = None,
        start_date: str = None,
        end_date: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from multiple database tables and return them as Pandas DataFrames.
    Handles large datasets by processing in chunks to avoid PostgreSQL synchronization issues.

    Parameters
    ----------
    table_query_mapping : Dict[str, str]
        A dictionary mapping table names to their respective SQL queries.
    logger : Any
        A logger instance for logging errors and execution info.
    tenant_id : str, optional
        The tenant ID to filter queries.
    start_date : str, optional
        Start date for filtering queries in format YYYY-MM-DD.
    end_date : str, optional
        End date for filtering queries in format YYYY-MM-DD.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are Pandas DataFrames
        containing the query results.

    Raises
    ------
    Exception
        If an error occurs while executing queries.
    """
    dataframes = {}

    session = get_database_session()
    try:
        tenant_info = f"for tenant {tenant_id}" if tenant_id else "for all tenants"
        date_info = f" with date range: {start_date} to {end_date}" if start_date and end_date else ""
        logger.info(f"Fetching data from database tables {tenant_info}{date_info}")

        for table_name, query in table_query_mapping.items():
            logger.info(f"Executing query for table: {table_name} {tenant_info}")

            # Prepare parameters for the query
            params = {}
            if tenant_id:
                params["tenant_id"] = tenant_id
            if start_date and end_date:
                params["start_date"] = start_date
                params["end_date"] = end_date
                logger.info(f"Applying date filter: {start_date} to {end_date}")

            # First get the total count using a simplified query
            total_count = _get_total_count(table_name=table_name,
                                           tenant_id=tenant_id,
                                           start_date=start_date,
                                           end_date=end_date,
                                           session=session,
                                           logger=logger)
            logger.info(f"Total records to fetch for {table_name} in date range {start_date} to {end_date} for tenant {tenant_id}: {total_count}")

            if total_count == 0:
                dataframes[table_name] = pd.DataFrame()
                continue

            # Get columns from first batch
            params["offset"] = 0
            params["limit"] = 1

            # Set chunk size with default value of 500 to avoid PostgreSQL synchronization issues
            # Reduced from 1000 to prevent "insufficient data in 'D' message" errors
            chunk_size = int(os.getenv("QUERY_CHUNK_SIZE", "2000"))
            all_rows = []
            offset = 0
            max_retries = 3
            retry_delay = 5

            while offset < total_count:
                params["offset"] = offset
                params["limit"] = chunk_size
                logger.info(
                    f"Fetching chunk {offset // chunk_size + 1} for {table_name} (offset: {offset}, limit: {chunk_size})")

                # Retry logic for each chunk with connection refresh
                # This handles PostgreSQL synchronization issues that can occur when fetching large datasets
                # Common errors include "insufficient data in 'D' message" and "lost synchronization"
                for attempt in range(max_retries):
                    try:
                        start_time = datetime.now()
                        # Add statement timeout to prevent long-running queries from hanging
                        # This helps prevent connection timeouts during large data fetches
                        session.execute(text("SET statement_timeout = 300000"))  # 5 minutes timeout
                        
                        # Execute the query with current parameters (offset and limit for chunking)
                        result = session.execute(text(query), params)
                        rows = result.fetchall()
                        columns = result.keys()
                        
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        # Success - break out of retry loop and continue to next chunk
                        break  
                    except Exception as e:
                        # Check if this is a PostgreSQL synchronization error that can be retried
                        # These errors are typically transient and can be resolved by refreshing the connection
                        if "insufficient data in" in str(e) or "lost synchronization" in str(e):
                            logger.warning(f"PostgreSQL synchronization error detected for chunk {offset // chunk_size + 1}: {str(e)}")
                            if attempt == max_retries - 1:
                                # Last attempt failed - re-raise the exception to stop execution
                                logger.error(f"All retry attempts failed for chunk {offset // chunk_size + 1}. Aborting.")
                                raise
                        elif attempt == max_retries - 1:
                            # Last attempt failed for a different error - re-raise the exception
                            logger.error(f"All retry attempts failed for chunk {offset // chunk_size + 1}: {str(e)}")
                            raise
                        
                        # Log retry attempt and wait before retrying
                        logger.warning(f"Attempt {attempt + 1} failed for chunk {offset // chunk_size + 1}: {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1} of {max_retries})")
                        time.sleep(retry_delay)
                        
                        # Refresh session for retry to prevent synchronization issues
                        # Close existing session (ignore errors if session is already closed)
                        try:
                            session.close()
                        except:
                            pass
                        # Create fresh session for retry attempt
                        session = get_database_session()
                        logger.info(f"Session refreshed for retry attempt {attempt + 2}")

                if not rows:
                    break

                all_rows.extend(rows)
                offset += chunk_size

                # Log progress with timing information
                progress = min(offset, total_count)
                logger.info(
                    f"Progress for {table_name}: {progress}/{total_count} records ({progress / total_count * 100:.1f}%) - Chunk processed in {duration:.2f} seconds")

            # Create DataFrame from all rows
            df = pd.DataFrame(all_rows, columns=columns)
            dataframes[table_name] = df
            logger.info(f"Successfully fetched {len(df)} rows from {table_name} {tenant_info}")

    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred while fetching data: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            session.close()
        except:
            pass

    return dataframes


def list_database_tables_with_columns(logger: Any) -> Dict[str, List[Dict[str, str]]]:
    """
    Get a list of all tables in the database with their column information.

    Parameters
    ----------
    logger : Any
        A logger instance for logging errors and execution info.

    Returns
    -------
    Dict[str, List[Dict[str, str]]]
        A dictionary where keys are table names and values are lists of column info.
        Each column info contains: column_name, data_type, is_nullable

    Raises
    ------
    Exception
        If an error occurs while querying the database.
    """
    session = get_database_session()
    try:
        logger.info("Fetching database tables with column information")

        # Query to get table and column information
        query = """
                SELECT table_name, \
                       column_name, \
                       data_type, \
                       is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position; \
                """

        result = session.execute(text(query))
        rows = result.fetchall()

        # Group columns by table
        tables_info = {}
        for row in rows:
            table_name, column_name, data_type, is_nullable = row
            if table_name not in tables_info:
                tables_info[table_name] = []

            tables_info[table_name].append({
                'column_name': column_name,
                'data_type': data_type,
                'is_nullable': is_nullable
            })

        logger.info(f"Found {len(tables_info)} tables in the database")
        for table_name, columns in tables_info.items():
            logger.info(f"Table '{table_name}' has {len(columns)} columns")

        return tables_info

    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred while fetching table information: {e}")
        raise
    finally:
        session.close()


def fetch_table_data(
        table_name: str,
        logger: Any,
        tenant_id: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = None
) -> pd.DataFrame:
    """
    Fetch all data from a specific database table and return it as a Pandas DataFrame.

    Parameters
    ----------
    table_name : str
        The name of the table to fetch data from.
    logger : Any
        A logger instance for logging errors and execution info.
    tenant_id : str, optional
        The tenant ID to filter queries if the table has a tenant_id column.
    start_date : str, optional
        Start date for filtering queries in format YYYY-MM-DD.
    end_date : str, optional
        End date for filtering queries in format YYYY-MM-DD.
    limit : int, optional
        Maximum number of records to fetch. If None, fetches all records.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the query results.

    Raises
    ------
    Exception
        If an error occurs while executing the query.
    """
    session = get_database_session()
    try:
        logger.info(f"Fetching data from table: {table_name}")

        # Build the base query
        query = f"SELECT * FROM {table_name}"
        params = {}
        conditions = []

        # Add tenant_id filter if provided
        if tenant_id:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        # Add date range filter if provided (assumes a common date column name)
        if start_date and end_date:
            # You might need to adjust the date column name based on your table structure
            date_column = "created_at"  # or "journal_date", "transaction_date", etc.
            conditions.append(f"{date_column} >= :start_date AND {date_column} <= :end_date")
            params["start_date"] = start_date
            params["end_date"] = end_date

        # Add WHERE clause if there are conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY clause for consistent results
        query += " ORDER BY id"

        # Add LIMIT if specified
        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit

        logger.info(f"Executing query: {query}")
        logger.info(f"Query parameters: {params}")

        # Execute query and fetch results
        result = session.execute(text(query), params)
        rows = result.fetchall()
        columns = result.keys()

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)

        logger.info(f"Successfully fetched {len(df)} rows from table '{table_name}'")
        logger.info(f"Table columns: {list(df.columns)}")

        return df

    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred while fetching data from table '{table_name}': {e}")
        raise
    finally:
        session.close()


def fetch_table_data_by_line_item_ids(
        table_name: str,
        tenant_id: str,
        line_item_id_list: List[str],
        logger: Any,
        limit: int = None
) -> pd.DataFrame:
    """
    Fetch data from a specific database table filtered by line_item_id list and return it as a Pandas DataFrame.
    Uses the SQL queries from gst_account_mapping_database_queries.yaml to construct the appropriate query.

    Parameters
    ----------
    table_name : str
        The name of the table to fetch data from.
    tenant_id : str
        The tenant ID to filter queries if the table has a tenant_id column.
    line_item_id_list : List[str]
        List of line_item_id values to filter the query.
    logger : Any
        A logger instance for logging errors and execution info.
    limit : int, optional
        Maximum number of records to fetch. If None, fetches all records.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the query results.

    Raises
    ------
    Exception
        If an error occurs while executing the query.
    """
    session = get_database_session()
    try:
        logger.info(f"Fetching data from table: {table_name} for {len(line_item_id_list)} line_item_ids")

        # Load the query from the YAML file
        import yaml
        import os
        import re

        yaml_path = os.path.join(os.path.dirname(__file__), 'gst_account_mapping_database_queries.yaml')
        with open(yaml_path, 'r') as file:
            queries = yaml.safe_load(file)

        if table_name not in queries:
            raise ValueError(f"Table '{table_name}' not found in database queries configuration")

        base_query = queries[table_name]

        # Add line_item_id filter (required)
        if not line_item_id_list:
            logger.warning("Empty line_item_id_list provided, returning empty DataFrame")
            return pd.DataFrame()

        # Clean the base query by removing trailing semicolon if present
        base_query = base_query.strip()
        if base_query.endswith(';'):
            base_query = base_query[:-1]

        # Remove the problematic DISTINCT ON clauses and pagination
        # These are causing line items to be filtered out

        # Fix for accrec_accpay: Remove DISTINCT ON (ili.invoice_id)
        if table_name == 'accrec_accpay':
            base_query = base_query.replace(
                'SELECT DISTINCT ON (ili.invoice_id)',
                'SELECT'
            )

        # Fix for cashrec_cashpaid: Remove DISTINCT ON (btli.bank_transaction_id)
        if table_name == 'cashrec_cashpaid':
            base_query = base_query.replace(
                'SELECT DISTINCT ON (btli.bank_transaction_id)',
                'SELECT'
            )

        # Remove pagination from the base query since we'll filter by line_item_id
        # Remove the entire paginated_data CTE and its usage
        base_query = re.sub(
            r',\s*paginated_data\s+AS\s+\([^)]+\)\s*\)\s*SELECT[^)]+FROM\s+paginated_data[^;]*',
            '',
            base_query,
            flags=re.DOTALL
        )

        # Also remove LIMIT and OFFSET from the base query
        base_query = re.sub(r'LIMIT\s+:\s*limit\s+OFFSET\s+:\s*offset', '', base_query)

        # Remove ORDER BY row_num since row_num won't exist anymore
        base_query = re.sub(r'ORDER\s+BY\s+row_num[^;]*', '', base_query)

        # Create placeholders for the IN clause
        placeholders = ', '.join([f':line_item_id_{i}' for i in range(len(line_item_id_list))])

        # Wrap the base query and add line_item_id filtering
        wrapped_query = f"""
        WITH base_data AS (
            {base_query}
        )
        SELECT * FROM base_data 
        WHERE line_item_id IN ({placeholders})
        """

        # Add LIMIT if specified
        if limit:
            wrapped_query += f" LIMIT {limit}"

        # Set up parameters - remove date filtering to ensure we get all records
        params = {
            'tenant_id': tenant_id,
            'start_date': '1900-01-01',  # Use a very early date to get all records
            'end_date': '2100-01-01',  # Use a very late date to get all records
        }

        # Add parameters for each line_item_id
        for i, line_item_id in enumerate(line_item_id_list):
            params[f'line_item_id_{i}'] = line_item_id

        logger.info(f"Executing wrapped query for table: {table_name}")
        logger.debug(f"Modified query: {wrapped_query}")

        # Execute query and fetch results
        result = session.execute(text(wrapped_query), params)
        rows = result.fetchall()
        columns = result.keys()

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)

        logger.info(f"Successfully fetched {len(df)} rows from table '{table_name}' for line_item_ids")
        if len(df) > 0:
            logger.info(f"Table columns: {list(df.columns)}")

        # Log missing line_item_ids for debugging
        if len(df) > 0:
            found_line_items = set(df['line_item_id'].unique())
            missing_line_items = set(line_item_id_list) - found_line_items
            if missing_line_items:
                logger.warning(
                    f"Missing {len(missing_line_items)} line_item_ids from table '{table_name}': {list(missing_line_items)[:10]}...")

        return df

    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred while fetching data from table '{table_name}' for line_item_ids: {e}")
        raise
    finally:
        session.close()


def register_data_into_local_storage(
        decomposed_tables: Dict[str, Dict[str, pd.DataFrame]],
        raw_dataset_dir: str,
        logger: Any
):
    """
    Save decomposed tenant-specific DataFrames as CSV files into local storage.

    Parameters
    ----------
    decomposed_tables : dict of {str: dict of {str: pd.DataFrame}}
        A nested dictionary mapping table names to tenant-specific DataFrames.
    raw_dataset_dir : str
        The directory where CSV files will be stored. Each table will have its own subdirectory.
    logger: Any

    Returns
    -------
    None
    """
    logger.info("Registering data into local storage...")
    for table_name, tenant_id_dataframes in decomposed_tables.items():
        table_name_dir = os.path.join(raw_dataset_dir, table_name)
        os.makedirs(table_name_dir, exist_ok=True)
        for tenant_id, tenant_dataframe in tenant_id_dataframes.items():
            tenant_dataframe_path = os.path.join(table_name_dir, f"{tenant_id}.csv")
            tenant_dataframe.to_csv(tenant_dataframe_path, index=False)
            logger.info(f"Saved tenant '{tenant_id}' data from table '{table_name}' to {tenant_dataframe_path}.")
    logger.info("Data registration into local storage completed.")


def data_collection_step(
        raw_dataset_dir: str,
        tenant_id: str
) -> None:
    """
    Execute the complete data collection process: fetch tenant IDs, fetch data for each tenant,
    and register the data into local storage.

    Parameters
    ----------
    raw_dataset_dir : str
        The base directory where the raw CSV files will be stored.
    tenant_id: str

    Returns
    -------
    None
    """
    logger.info(f"Starting data collection step for tenant: {tenant_id}")

    # Read training data date range from environment variables
    start_date = os.getenv('TRAINING_DATA_START_DATE')
    end_date = os.getenv('TRAINING_DATA_END_DATE')

    if start_date and end_date:
        logger.info(f"Using date range from environment variables: {start_date} to {end_date}")
    else:
        logger.info(
            "No date range specified in environment variables TRAINING_DATA_START_DATE and TRAINING_DATA_END_DATE")

    logger.info("Loading the database queries")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_queries_path = os.path.join(script_dir, 'gst_account_mapping_database_queries.yaml')
    with open(database_queries_path, 'r') as file:
        table_query_mapping = yaml.safe_load(file)

    # Initialize decomposed_tables structure
    decomposed_tables = {table_name: {} for table_name in table_query_mapping.keys()}

    # Fetch data for this tenant
    tenant_tables = fetch_database_data(
        table_query_mapping=table_query_mapping,
        logger=logger,
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date
    )

    # Add the tenant's data to the decomposed_tables structure
    for table_name, df in tenant_tables.items():
        if not df.empty:
            decomposed_tables[table_name][tenant_id] = df
            logger.info(f"Added {len(df)} records for tenant {tenant_id} to table {table_name}")

    decomposed_tables = {
        table_name: tenant_table_dict
        for table_name, tenant_table_dict in decomposed_tables.items() if tenant_table_dict
    }
    logger.info("Data fetching completed for all tenants.")

    # excluding test samples from raw data
    tenant_test_df = fetch_table_data(table_name='gst_account_validation_eval_set',
                                      logger=logger,
                                      tenant_id=tenant_id)
    if len(tenant_test_df) > 0 and decomposed_tables:
        for table_name, tenant_table_dict in decomposed_tables.items():
            test_line_item_ids = tenant_test_df[
                tenant_test_df['table_name'] == table_name
            ]['line_item_id'].unique().tolist()
            table_df = tenant_table_dict[tenant_id]
            table_df = table_df[~table_df['line_item_id'].isin(test_line_item_ids)]
            table_df.reset_index(drop=True, inplace=True)
            decomposed_tables[table_name][tenant_id] = table_df

    # Save the data to local storage
    register_data_into_local_storage(
        decomposed_tables=decomposed_tables,
        raw_dataset_dir=raw_dataset_dir,
        logger=logger
    )

    logger.info("Data collection step completed successfully.")


def tenant_table_data_collection_step(
        tenant_id: str,
        table_name: str,
        start_date: str,
        end_date: str,
        raw_dataset_dir: str,
        logger: logging.Logger,
):
    logger.info(f"Collecting data for tenant: {tenant_id} from {start_date} to {end_date}...")
    logger.info("Loading the database queries")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_queries_path = os.path.join(script_dir, 'gst_account_mapping_database_queries.yaml')
    with open(database_queries_path, 'r') as file:
        table_query_mapping = yaml.safe_load(file)
    table_query_mapping = {table_name: table_query_mapping[table_name]}

    table2df = fetch_database_data(
        table_query_mapping=table_query_mapping,
        logger=logger,
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date
    )
    decomposed_tables = {table_name: {tenant_id: table2df[table_name]}}

    # Save the data to local storage
    register_data_into_local_storage(
        decomposed_tables=decomposed_tables,
        raw_dataset_dir=raw_dataset_dir,
        logger=logger
    )


def tenant_data_collection(
        tenant_id: str,
        start_date: str,
        end_date: str,
        logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    logger.info(f"Collecting data for tenant: {tenant_id} from {start_date} to {end_date}...")
    logger.info("Loading the database queries")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_queries_path = os.path.join(script_dir, 'gst_account_mapping_database_queries.yaml')
    with open(database_queries_path, 'r') as file:
        table_query_mapping = yaml.safe_load(file)

    table2df = fetch_database_data(
        table_query_mapping=table_query_mapping,
        logger=logger,
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date
    )
    return table2df


def get_latest_timestamp(timestamps: List[str]) -> str:
    """
    Get the latest (most recent) timestamp from a list of formatted timestamp strings.

    Parameters
    ----------
    timestamps : List[str]
        A list of timestamp strings in the format 'YYYY-MM-DD-HH-MM-SS'.

    Returns
    -------
    str
        The latest (most recent) timestamp string.

    Raises
    ------
    ValueError
        If the timestamps list is empty.
    """
    if not timestamps:
        raise ValueError("The list of timestamps is empty.")

    # Convert each timestamp string to a datetime object and use it as a key for max()
    latest_timestamp = max(
        timestamps,
        key=lambda ts: datetime.strptime(ts, '%Y-%m-%d-%H-%M-%S')
    )
    return latest_timestamp


def model_collection_step(
        base_data_dir: str,
        bucket_name: str,
        checkpoints_prefix: str,
        checkpoints_dir: str,
        logger: logging.Logger,
        model_version: str = None,
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Collect models and related artifacts from S3 and return a dictionary containing the model data.

    This function performs the following steps:
      1. Retrieves the S3 client and necessary environment variables.
      2. Lists all tenant directories under the specified S3 bucket prefix.
      3. For each tenant:
         a. Lists the model version directories.
         b. Selects the latest model version based on the timestamp.
         c. Downloads the model artifacts from S3 to a local directory.
         d. Loads the model, encoders, hyperparameters, and training data from the downloaded files.
      4. Aggregates the data into a dictionary keyed by tenant IDs.

    Parameters
    ----------
    base_data_dir: str
    bucket_name: str
    checkpoints_prefix: str
    checkpoints_dir: str
    logger : logging.Logger
        Logger object to track the steps in the model collection process.
    model_version: str, optional

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary containing the model data with the following keys:
            - "name": The name of the model (common across tenants).
            - "version": The version of the model (common across tenants).
            - "models": A dictionary where each key is a tenant ID and the value is a dictionary with:
                - "encoders": The loaded encoders.
                - "hyperparameters": The hyperparameters dictionary.
                - "train_df": The training DataFrame.
                - "knn_model": The loaded KNN model.
                - "pca_model": The PCA model for dimensionality reduction (if exists).
                - "embeddings": The loaded text embeddings (if any).
    """
    logger.info("Starting model collection process.")

    s3_client = get_s3_client()

    model_data = {
        "name": None,
        "version": None,
        "models": {}
    }

    logger.info("Listing tenant directories from S3 using prefix: %s", checkpoints_prefix)
    tenant_id_list = list_s3_directories(s3_client=s3_client,
                                         bucket_name=bucket_name,
                                         prefix=checkpoints_prefix)
    logger.info("Found tenant directories: %s", tenant_id_list)

    for tenant_id in tenant_id_list:
        model_data["models"][tenant_id] = {}
        logger.info("Fetching model for tenant: %s", tenant_id)
        tenant_prefix = os.path.join(checkpoints_prefix, tenant_id)
        logger.info("Listing model versions for tenant %s using prefix: %s", tenant_id, tenant_prefix)
        if not model_version:
            # extracting the latest model version
            model_versions = list_s3_directories(s3_client=s3_client,
                                                 bucket_name=bucket_name,
                                                 prefix=tenant_prefix)
            logger.info("Found model versions for tenant %s: %s", tenant_id, model_versions)

            model_version = get_latest_timestamp(timestamps=model_versions)
        logger.info("Latest model version for tenant %s: %s", tenant_id, model_version)

        tenant_id_checkpoint_dir = os.path.join(base_data_dir,
                                                checkpoints_dir,
                                                tenant_id,
                                                model_version)
        logger.info("S3 artifacts for tenant %s will be downloaded from the prefix: %s to local directory: %s",
                    tenant_id, os.path.join(tenant_prefix, model_version), tenant_id_checkpoint_dir)

        table_models_list = list_s3_directories(s3_client=s3_client,
                                                bucket_name=bucket_name,
                                                prefix=os.path.join(tenant_prefix, model_version))
        for table_name in table_models_list:
            logger.info(f"Downloading model data for the {table_name} table from "
                        f"s3://{bucket_name}/{os.path.join(tenant_prefix, model_version, table_name)}...")
            tenant_id_table_checkpoint_dir = os.path.join(base_data_dir,
                                                          checkpoints_dir,
                                                          tenant_id,
                                                          model_version,
                                                          table_name)
            download_s3_directory(s3_client=s3_client,
                                  bucket_name=bucket_name,
                                  prefix=os.path.join(tenant_prefix, model_version, table_name),
                                  local_data_dir=tenant_id_table_checkpoint_dir)

        for table_name in table_models_list:
            logger.info("Loading model artifacts for tenant: %s", tenant_id)
            tenant_id_table_checkpoint_dir = os.path.join(base_data_dir,
                                                          checkpoints_dir,
                                                          tenant_id,
                                                          model_version,
                                                          table_name)

            with open(os.path.join(tenant_id_table_checkpoint_dir, 'encoders.pkl'), 'rb') as handle:
                encoders = pickle.load(handle)
            with open(os.path.join(tenant_id_table_checkpoint_dir, 'hyperparameters.json'), 'r') as handle:
                hyperparameters = json.load(handle)
            with open(os.path.join(tenant_id_table_checkpoint_dir, 'model.pkl'), 'rb') as handle:
                knn_model = pickle.load(handle)
            train_df = pd.read_csv(os.path.join(tenant_id_table_checkpoint_dir, 'train.csv'))
            raw_df = pd.read_csv(os.path.join(tenant_id_table_checkpoint_dir, 'raw.csv'))
            logger.info("Successfully loaded model artifacts for tenant: %s", tenant_id)

            if model_data["name"] is None:
                model_data["name"] = hyperparameters["model_name"]
                logger.info("Set model name to: %s", model_data["name"])
            if model_data["version"] is None:
                model_data["version"] = hyperparameters["version"]
                logger.info("Set model version to: %s", model_data["version"])

            model_data["models"][tenant_id][table_name] = {
                "encoders": encoders,
                "hyperparameters": hyperparameters,
                "train_df": train_df,
                "raw_df": raw_df,
                "knn_model": knn_model
            }
            logger.info("Loaded model for tenant: %s", tenant_id)

    logger.info("Completed model collection process.")
    return model_data


def llm_context_collection_step(
        base_data_dir: str,
        bucket_name: str,
        checkpoints_prefix: str,
        checkpoints_dir: str,
        logger: logging.Logger,
        context_version: str = None,
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Collect and load LLM context data for account mapping inference from S3.

    This function downloads the latest model artifacts for each tenant from S3,
    loads the required context data including feature matrices, tokenizers,
    hyperparameters, and datasets, and organizes them for LLM-based inference.

    Parameters
    ----------
    base_data_dir : str
        The base directory path where downloaded model artifacts will be stored locally.
    bucket_name : str
        The name of the S3 bucket containing the model checkpoints and artifacts.
    checkpoints_prefix : str
        The S3 prefix (folder path) where tenant model checkpoints are stored.
    checkpoints_dir : str
        The local directory name within base_data_dir where checkpoints will be saved.
    logger : logging.Logger
        Logger instance for tracking the collection process and debugging.
    context_version: str, optional

    Returns
    -------
    Dict[str, Union[str, Dict[str, Any]]]
        Dictionary containing LLM context data with the following structure:
        - "version": str or None, the model version extracted from hyperparameters
        - "context": Dict[str, Dict[str, Any]], nested dictionary where:
            - First level keys are tenant IDs
            - Second level keys are table names
            - Values contain model artifacts: feature_token_matrix, hyperparameters,
              feature_tokenizer, df (training data), and raw_df (raw data)
    """
    logger.info("Starting the LLM context collection process.")

    s3_client = get_s3_client()
    llm_context_data = {
        "version": None,
        "context": {}
    }

    logger.info("Listing tenant directories from S3 using prefix: %s", checkpoints_prefix)
    tenant_id_list = list_s3_directories(s3_client=s3_client,
                                         bucket_name=bucket_name,
                                         prefix=checkpoints_prefix)
    logger.info("Found tenant directories: %s", tenant_id_list)

    for tenant_id in tenant_id_list:
        llm_context_data["context"][tenant_id] = {}
        logger.info("Fetching LLM context for tenant: %s", tenant_id)
        tenant_prefix = os.path.join(checkpoints_prefix, tenant_id)
        logger.info("Listing LLM context versions for tenant %s using prefix: %s", tenant_id, tenant_prefix)
        if not context_version:
            # extracting the latest context version
            context_versions = list_s3_directories(s3_client=s3_client,
                                                   bucket_name=bucket_name,
                                                   prefix=tenant_prefix)
            logger.info("Found LLM context versions for tenant %s: %s", tenant_id, context_versions)

            context_version = get_latest_timestamp(timestamps=context_versions)
        logger.info("Latest model version for tenant %s: %s", tenant_id, context_version)

        tenant_id_checkpoint_dir = os.path.join(base_data_dir,
                                                checkpoints_dir,
                                                tenant_id,
                                                context_version)
        logger.info("S3 artifacts for tenant %s will be downloaded from the prefix: %s to local directory: %s",
                    tenant_id, os.path.join(tenant_prefix, context_version), tenant_id_checkpoint_dir)

        table_models_list = list_s3_directories(s3_client=s3_client,
                                                bucket_name=bucket_name,
                                                prefix=os.path.join(tenant_prefix, context_version))
        for table_name in table_models_list:
            logger.info(f"Downloading model data for the {table_name} table from "
                        f"s3://{bucket_name}/{os.path.join(tenant_prefix, context_version, table_name)}...")
            tenant_id_table_checkpoint_dir = os.path.join(base_data_dir,
                                                          checkpoints_dir,
                                                          tenant_id,
                                                          context_version,
                                                          table_name)
            download_s3_directory(s3_client=s3_client,
                                  bucket_name=bucket_name,
                                  prefix=os.path.join(tenant_prefix, context_version, table_name),
                                  local_data_dir=tenant_id_table_checkpoint_dir)

        for table_name in table_models_list:
            logger.info("Loading LLM context for tenant: %s", tenant_id)
            tenant_id_table_checkpoint_dir = os.path.join(base_data_dir,
                                                          checkpoints_dir,
                                                          tenant_id,
                                                          context_version,
                                                          table_name)

            with open(os.path.join(tenant_id_table_checkpoint_dir, 'feature_token_matrix.pkl'), 'rb') as handle:
                feature_token_matrix = pickle.load(handle)
            with open(os.path.join(tenant_id_table_checkpoint_dir, 'feature_tokenizer.pkl'), 'rb') as handle:
                feature_tokenizer = pickle.load(handle)
            with open(os.path.join(tenant_id_table_checkpoint_dir, 'hyperparameters.json'), 'r') as handle:
                hyperparameters = json.load(handle)
            df = pd.read_csv(os.path.join(tenant_id_table_checkpoint_dir, 'train.csv'))
            raw_df = pd.read_csv(os.path.join(tenant_id_table_checkpoint_dir, 'raw.csv'))
            logger.info("Successfully loaded LLM context for tenant: %s", tenant_id)

            if llm_context_data["version"] is None:
                llm_context_data["version"] = hyperparameters["version"]
                logger.info("Set model version to: %s", llm_context_data["version"])

            llm_context_data["context"][tenant_id][table_name] = {
                "feature_token_matrix": feature_token_matrix,
                "hyperparameters": hyperparameters,
                "feature_tokenizer": feature_tokenizer,
                "df": df,
                "raw_df": raw_df,
            }
            logger.info("Loaded model for tenant: %s", tenant_id)

    logger.info("Completed model collection process.")
    return llm_context_data


def model_selection_artifacts_collection_step(
        base_data_dir: str,
        bucket_name: str,
        model_selection_prefix: str,
        model_selection_dir: str,
        logger: logging.Logger
):
    """
    Collects model selection artifacts for multiple tenants from an S3 bucket and organizes them locally.

    Parameters
    ----------
    base_data_dir : str
        The base directory on the local filesystem where artifacts are organized.
    bucket_name : str
        The name of the S3 bucket from which the artifacts are fetched.
    model_selection_prefix : str
        The prefix path in the S3 bucket to locate model selection data.
    model_selection_dir : str
        The subdirectory under `base_data_dir` where model selection artifacts are saved.
    logger : logging.Logger
        A logger instance used for logging messages during the process.

    Returns
    -------
    dict
        A configuration dictionary containing:
          - 'name': The name of the model.
          - 'version': The version of the model.
          - 'model_selection': A nested dictionary with tenant-specific and table-specific artifacts
            containing hyperparameters, cross-validation results, and best configurations.

    Notes
    -----
    - This function interacts with S3 to download and organize data locally.
    - Relies on utility functions for S3 operations as well as JSON parsing to extract and store relevant information.
    """
    logger.info("Starting model selection artifacts collection process.")
    config = {
        "name": None,
        "version": None,
        "model_selection": {}
    }
    s3_client = get_s3_client()
    tenant_id_list = list_s3_directories(s3_client=s3_client,
                                         bucket_name=bucket_name,
                                         prefix=model_selection_prefix)
    if tenant_id_list is None:
        logger.warning("No tenant directories found in S3 bucket %s using prefix %s",
                       bucket_name, model_selection_prefix)
        return config

    logger.debug("Found tenant directories: %s", tenant_id_list)
    for tenant_id in tenant_id_list:
        config["model_selection"][tenant_id] = {}
        logger.debug("Processing tenant: %s", tenant_id)
        tenant_prefix = os.path.join(model_selection_prefix, tenant_id)
        logger.debug("Listing model selection versions for tenant %s using prefix: %s", tenant_id, tenant_prefix)
        model_selection_versions = list_s3_directories(s3_client=s3_client,
                                                       bucket_name=bucket_name,
                                                       prefix=tenant_prefix)
        logger.debug("Found model versions for tenant %s: %s", tenant_id, model_selection_versions)
        latest_model_selection_version = get_latest_timestamp(timestamps=model_selection_versions)
        logger.debug("Latest model version for tenant %s: %s", tenant_id, latest_model_selection_version)
        tenant_id_model_selection_dir = os.path.join(base_data_dir,
                                                     model_selection_dir,
                                                     tenant_id,
                                                     latest_model_selection_version)
        logger.debug("Downloading S3 artifacts for tenant %s from prefix: %s to local directory: %s",
                     tenant_id, os.path.join(tenant_prefix, latest_model_selection_version), tenant_id_model_selection_dir)

        artifacts_table_list = list_s3_directories(s3_client=s3_client,
                                                   bucket_name=bucket_name,
                                                   prefix=os.path.join(tenant_prefix, latest_model_selection_version))
        for table_name in artifacts_table_list:
            logger.info(f"Downloading model selection data for the {table_name} table from "
                        f"s3://{bucket_name}/{os.path.join(tenant_prefix, latest_model_selection_version, table_name)}...")
            tenant_id_table_artifacts_dir = os.path.join(base_data_dir,
                                                         model_selection_dir,
                                                         tenant_id,
                                                         latest_model_selection_version,
                                                         table_name)
            download_s3_directory(s3_client=s3_client,
                                  bucket_name=bucket_name,
                                  prefix=os.path.join(tenant_prefix, latest_model_selection_version, table_name),
                                  local_data_dir=tenant_id_table_artifacts_dir)
        for table_name in artifacts_table_list:
            logger.debug("Loading model artifacts for tenant: %s", tenant_id)
            tenant_id_table_artifacts_dir = os.path.join(base_data_dir,
                                                         model_selection_dir,
                                                         tenant_id,
                                                         latest_model_selection_version,
                                                         table_name)
            with open(os.path.join(tenant_id_table_artifacts_dir, 'cv_results.json'), 'r') as handle:
                cv_results = json.load(handle)
            with open(os.path.join(tenant_id_table_artifacts_dir, 'cv_best_minimal_config.json'), 'r') as handle:
                cv_best_minimal_config = json.load(handle)
            with open(os.path.join(tenant_id_table_artifacts_dir, 'hyperparameters.json'), 'r') as handle:
                hyperparameters = json.load(handle)
            logger.debug("Successfully loaded model artifacts for tenant: %s", tenant_id)
            if config["name"] is None:
                config["name"] = hyperparameters["model_name"]
                logger.debug("Set model name to: %s", config["name"])
            if config["version"] is None:
                config["version"] = hyperparameters["version"]
                logger.debug("Set model version to: %s", config["version"])
            config["model_selection"][tenant_id][table_name] = {
                "hyperparameters": hyperparameters,
                "cv_results": cv_results,
                "cv_best_minimal_config": cv_best_minimal_config
            }
            logger.debug("Loaded model selection data for tenant: %s", tenant_id)
    logger.info("Completed model selection data collection process.")
    return config


def _get_total_count(
        table_name: str,
        tenant_id: str,
        start_date: str,
        end_date: str,
        session,
        logger: logging.Logger
) -> int:
    """
    Calculate the total count of records based on specific table and query criteria.

    Parameters
    ----------
    table_name : str
        The name of the table to query. Supported values include 'accrec_accpay',
        'cashrec_cashpaid', 'expclaim', 'accrec_accpay_payments', and 'accreccredit_accpaycredit'.
    tenant_id : str
        The tenant identifier to filter the query for specific tenants.
    start_date : str
        The start date for filtering records (format: YYYY-MM-DD).
    end_date : str
        The end date for filtering records (format: YYYY-MM-DD).
    session : Session
        The database session used to execute queries.
    logger : logging.Logger
        A logger instance used for logging errors and events.

    Returns
    -------
    int
        The total count of records matching the query criteria.

    Raises
    ------
    ValueError
        If an unsupported table_name is provided.
    """
    params = {
        "tenant_id": tenant_id,
        "start_date": start_date,
        "end_date": end_date
    }

    if table_name == 'accrec_accpay':
        count_query = """
                      SELECT COUNT(*) FROM (
                                               SELECT ili.id
                                               FROM invoices i
                                                        JOIN invoices_line_items ili ON ili.invoice_unique_id = i.id
                                               WHERE i.tenant_id = :tenant_id
                                                 AND i.date BETWEEN :start_date AND :end_date
                                           ) as total_count; \
                      """
    elif table_name == 'cashrec_cashpaid':
        count_query = """
                      SELECT COUNT(*) FROM (
                                               SELECT btli.id
                                               FROM bank_transactions bt
                                                        JOIN bank_transactions_line_items btli ON btli.bank_transaction_unique_id = bt.id
                                               WHERE bt.tenant_id = :tenant_id
                                                 AND bt.date BETWEEN :start_date AND :end_date
                                                 AND (bt.type LIKE '%SPEND%' OR bt.type LIKE '%RECEIVE%')
                                           ) as total_count; \
                      """
    elif table_name == 'expclaim':
        count_query = """
                      SELECT COUNT(*) FROM (
                                               SELECT rli.id
                                               FROM expense_claims ec
                                                        JOIN expense_claim_receipts r ON r.expense_claim_unique_id = ec.id
                                                        JOIN expense_claim_receipt_line_items rli ON rli.receipt_unique_id = r.id
                                               WHERE ec.tenant_id = :tenant_id
                                                 AND ec.updated_date_utc BETWEEN :start_date AND :end_date
                                           ) as total_count; \
                      """
    elif table_name == 'accreccredit_accpaycredit':
        count_query = """
                      SELECT COUNT(*) FROM (
                                               SELECT cnli.id
                                               FROM credit_notes cn
                                                        JOIN credit_note_line_items cnli ON cnli.credit_note_unique_id = cn.id
                                               WHERE cn.tenant_id = :tenant_id
                                                 AND cn.date BETWEEN :start_date AND :end_date
                                           ) as total_count; \
                      """
    # elif table_name == 'accrec_accpay_payments':
    #     count_query = """
    #         SELECT COUNT(*) FROM (
    #             SELECT DISTINCT jl.journal_line_id
    #             FROM journal_lines jl
    #             INNER JOIN journals j ON jl.journal_id = j.journal_id
    #             WHERE jl.tenant_id = :tenant_id
    #                 AND j.journal_date BETWEEN :start_date AND :end_date
    #                 AND j.source_type IN ('ACCRECPAYMENT', 'ACCPAYPAYMENT')
    #         ) as total_count;
    #     """
    else:
        raise ValueError(f"Unknown table name: {table_name}")

    try:
        result = session.execute(text(count_query), params).scalar()
        return result if result is not None else 0
    except Exception as e:
        logger.error(f"Error getting count for {table_name}: {str(e)}")
        return 0


def load_tenant_table_data(
        tenant_id: str,
        table_name: str,
        raw_table_dir: str,
        table_columns: Dict[str, List[str]],
        table_categorical_features: Dict[str, List[str]],
        table_numerical_features: Dict[str, List[str]],
        table_target_variables: Dict[str, str],
        table_date_columns: Dict[str, str],
        table_filter_low_frequency_class_counts: Dict[str, bool],
        table_n_neighbors: Optional[Dict[str, int]] = None,
        table_knn_weight_modes: Optional[Dict[str, str]] = None,
) -> Dict:
    csv_path = os.path.join(raw_table_dir, f'{tenant_id}.csv')
    raw_df = pd.read_csv(csv_path)

    # validating input configs
    columns = table_columns[table_name]
    numerical_features = table_numerical_features[table_name]
    categorical_features = table_categorical_features[table_name]
    target_variable = table_target_variables[table_name]
    date_column = table_date_columns[table_name]
    filter_low_frequency_class_counts = table_filter_low_frequency_class_counts[table_name]

    if table_n_neighbors:
        n_neighbors = table_n_neighbors[table_name]
    else:
        n_neighbors = None
    if table_knn_weight_modes:
        knn_weight_mode = table_knn_weight_modes[table_name]
    else:
        knn_weight_mode = None

    validate_model_config(raw_df=raw_df,
                          categorical_features=categorical_features,
                          numerical_features=numerical_features,
                          columns=columns,
                          target_variable=target_variable,
                          date_column=date_column)
    table_data = {
        "csv_path": csv_path,
        "raw_df": raw_df,
        "columns": columns,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "target_variable": target_variable,
        "date_column": date_column,
        "filter_low_frequency_class_counts": filter_low_frequency_class_counts,
        "n_neighbors": n_neighbors,
        "knn_weight_mode": knn_weight_mode,
    }
    return table_data


def main():
    from ai.utils.initialization import logger
    # BASE_DATA_DIR = os.getenv("BASE_DATA_DIR")
    # GST_VALIDATION_DATA_DIR = os.getenv("GST_VALIDATION_DATA_DIR")
    # raw_dataset_dir = os.path.join(BASE_DATA_DIR, GST_VALIDATION_DATA_DIR)
    # os.makedirs(raw_dataset_dir, exist_ok=True)

    ACCOUNT_MAPPING_BASE_DATA_DIR = os.getenv('ACCOUNT_MAPPING_BASE_DATA_DIR')
    ACCOUNT_MAPPING_DATA_DIR = os.getenv('ACCOUNT_MAPPING_DATA_DIR')
    # ACCOUNT_MAPPING_MODEL_SELECTION_DIR = os.getenv('ACCOUNT_MAPPING_MODEL_SELECTION_DIR')
    ACCOUNT_MAPPING_CHECKPOINTS_DIR = os.getenv('ACCOUNT_MAPPING_CHECKPOINTS_DIR')
    ACCOUNT_MAPPING_LOGS_DIR = os.getenv('ACCOUNT_MAPPING_LOGS_DIR')
    ACCOUNT_MAPPING_BUCKET_NAME = os.getenv('ACCOUNT_MAPPING_BUCKET_NAME')

    llm_context_collection_step(
        base_data_dir=ACCOUNT_MAPPING_BASE_DATA_DIR,
        bucket_name=ACCOUNT_MAPPING_BUCKET_NAME,
        checkpoints_prefix=ACCOUNT_MAPPING_CHECKPOINTS_DIR,
        checkpoints_dir=ACCOUNT_MAPPING_CHECKPOINTS_DIR,
        logger=logger
    )
    # data_collection_step(
    #     raw_dataset_dir=raw_dataset_dir,
    #     logger=logger
    # )
    #
    # df = tenant_data_collection(
    #     tenant_id="77b67978-ae76-4480-ac96-066ea269541e",
    #     table_name="xxx",
    #     start_date="2024-01-01",
    #     end_date="2024-12-30",
    #     logger=logger
    # )


if __name__ == "__main__":
    main()
