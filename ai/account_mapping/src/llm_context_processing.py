import os
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

from common.logs import get_logger
from ai.utils.ecs_util import set_service_desired_count_to_zero
from ai.utils.config import load_llm_config
from ai.agents.jouranl_attr_classifier.llm_context_preprocessing import context_preprocessing_step
from ai.ml.knn.evaluation import evaluation_step

# Load environment variables
load_dotenv(override=True)
logger = get_logger()


async def main():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    # Reading configurations
    ACCOUNT_MAPPING_BASE_DATA_DIR = os.getenv('ACCOUNT_MAPPING_BASE_DATA_DIR')
    ACCOUNT_MAPPING_DATA_DIR = os.getenv('ACCOUNT_MAPPING_DATA_DIR')
    ACCOUNT_MAPPING_CHECKPOINTS_DIR = os.getenv('ACCOUNT_MAPPING_CHECKPOINTS_DIR')
    ACCOUNT_MAPPING_LOGS_DIR = os.getenv('ACCOUNT_MAPPING_LOGS_DIR')
    ACCOUNT_MAPPING_BUCKET_NAME = os.getenv('ACCOUNT_MAPPING_BUCKET_NAME')

    # Check that environment variables are not None
    if not all([ACCOUNT_MAPPING_BASE_DATA_DIR,
                ACCOUNT_MAPPING_DATA_DIR,
                ACCOUNT_MAPPING_CHECKPOINTS_DIR,
                ACCOUNT_MAPPING_LOGS_DIR,
                ACCOUNT_MAPPING_BUCKET_NAME]):
        logger.error("One or more required environment variables are not set.")
        raise ValueError

    # Defining the default features with their types and default model's hyperparameters
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.yaml")
    with open(config_path, "r") as file:
        default_config = yaml.safe_load(file)
    logger.info("Default configurations Loaded:")

    llm_config = load_llm_config(default_config=default_config,
                                 logger=logger)

    # ============ Context processing Step ============
    context_preprocessing_step(
        bucket_name=ACCOUNT_MAPPING_BUCKET_NAME,
        run_timestamp=run_timestamp,
        base_data_dir=ACCOUNT_MAPPING_BASE_DATA_DIR,
        raw_data_prefix=ACCOUNT_MAPPING_DATA_DIR,
        checkpoints_dir=ACCOUNT_MAPPING_CHECKPOINTS_DIR,
        logs_dir=ACCOUNT_MAPPING_LOGS_DIR,
        table_features=llm_config["table_features"],
        table_constraint_features=llm_config["table_constraint_features"],
        table_target_variables=llm_config["table_target_variables"],
        table_date_columns=llm_config["table_date_columns"],
        table_id_columns=llm_config["table_id_columns"],
        table_n_neighbors=llm_config["table_n_neighbors"],
    )

    # ============ Evaluation Step ============
    await evaluation_step(
        task_name="ACCOUNT_MAPPING_VALIDATION",
        method_name="LLM",
        method_run_timestamp=run_timestamp,
        base_data_dir=ACCOUNT_MAPPING_BASE_DATA_DIR,
        bucket_name=ACCOUNT_MAPPING_BUCKET_NAME,
        checkpoints_prefix=ACCOUNT_MAPPING_CHECKPOINTS_DIR,
        logger=logger
    )

    set_service_desired_count_to_zero()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
