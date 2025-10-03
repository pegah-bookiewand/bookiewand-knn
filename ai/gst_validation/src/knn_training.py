import os
import datetime
import yaml

from dotenv import load_dotenv

from ai.utils.initialization import logger
from ai.utils.data_collection import model_selection_artifacts_collection_step
from ai.utils.config import load_train_config
from ai.ml.knn.train import train_step
from ai.ml.knn.evaluation import evaluation_step

# Load environment variables
load_dotenv(override=True)


async def main():
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    # Reading configurations
    GST_VALIDATION_BASE_DATA_DIR = os.getenv('GST_VALIDATION_BASE_DATA_DIR')
    GST_VALIDATION_DATA_DIR = os.getenv('GST_VALIDATION_DATA_DIR')
    GST_VALIDATION_MODEL_SELECTION_DIR = os.getenv('GST_VALIDATION_MODEL_SELECTION_DIR')
    GST_VALIDATION_CHECKPOINTS_DIR = os.getenv('GST_VALIDATION_CHECKPOINTS_DIR')
    GST_VALIDATION_LOGS_DIR = os.getenv('GST_VALIDATION_LOGS_DIR')
    GST_VALIDATION_BUCKET_NAME = os.getenv('GST_VALIDATION_BUCKET_NAME')

    # Check that environment variables are not None
    if not all([GST_VALIDATION_BASE_DATA_DIR,
                GST_VALIDATION_DATA_DIR, 
                GST_VALIDATION_CHECKPOINTS_DIR, 
                GST_VALIDATION_LOGS_DIR,
                GST_VALIDATION_BUCKET_NAME]):
        logger.error("One or more required environment variables are not set.")
        raise ValueError

    # loading the features and model's hyperparameters of the model selection step
    model_selection_config = model_selection_artifacts_collection_step(
        base_data_dir=GST_VALIDATION_BASE_DATA_DIR,
        bucket_name=GST_VALIDATION_BUCKET_NAME,
        model_selection_prefix=GST_VALIDATION_MODEL_SELECTION_DIR,
        model_selection_dir=GST_VALIDATION_MODEL_SELECTION_DIR,
        logger=logger
    )

    # Defining the default features with their types and default model's hyperparameters
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.yaml")
    with open(config_path, "r") as file:
        default_config = yaml.safe_load(file)
    logger.info("Default configurations Loaded:")

    train_config = load_train_config(model_selection_config=model_selection_config,
                                     default_config=default_config,
                                     logger=logger)

    # ============ Training Step ============
    train_step(
        bucket_name=GST_VALIDATION_BUCKET_NAME,
        run_timestamp=run_timestamp,
        base_data_dir=GST_VALIDATION_BASE_DATA_DIR,
        raw_data_prefix=GST_VALIDATION_DATA_DIR,
        checkpoints_dir=GST_VALIDATION_CHECKPOINTS_DIR,
        logs_dir=GST_VALIDATION_LOGS_DIR,
        table_columns=train_config["table_columns"],
        table_categorical_features=train_config["table_categorical_features"],
        table_numerical_features=train_config["table_numerical_features"],
        table_target_variables=train_config["table_target_variables"],
        table_date_columns=train_config["table_date_columns"],
        table_n_neighbors=train_config["table_n_neighbors"],
        table_knn_weight_modes=train_config["table_knn_weight_modes"],
        table_filter_low_frequency_class_counts=train_config["table_filter_low_frequency_class_counts"],
        table_model_selection_version=train_config["table_model_selection_version"],
        random_seed=train_config["random_seed"],
    )

    # ============ Evaluation Step ============
    await evaluation_step(
        task_name="GST_VALIDATION",
        method_name="KNN",
        method_run_timestamp=run_timestamp,
        base_data_dir=GST_VALIDATION_BASE_DATA_DIR,
        bucket_name=GST_VALIDATION_BUCKET_NAME,
        checkpoints_prefix=GST_VALIDATION_CHECKPOINTS_DIR,
        logger=logger
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
