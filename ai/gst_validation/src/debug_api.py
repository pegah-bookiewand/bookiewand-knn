import os
import logging
from enum import Enum
from typing import Tuple, List, Optional

from bookiewand.enums.enums import TaxType
from bookiewand.util.data_helper import filter_gst_anomalies
import pandas as pd

from ai.utils.data_types import JournalLineRecord, JournalLinePredictionRequest, JournalLinePredictionResponse, JournalLinePredictionResult
from ai.utils.data_collection import tenant_data_collection
from common.logs import get_logger
from ai.gst_validation.src.api import (
    get_model_version,
    refresh_model,
    gst_validation,
    llm_gst_validation
)


class TaxName(str, Enum):
    GST_ON_INCOME = "GST on Income"
    GST_ON_EXPENSES = "GST on Expenses"
    GST_FREE_EXPENSES = "GST Free Expenses"
    GST_FREE_INCOME = "GST Free Income"
    BAS_EXCLUDED = "BAS Excluded"
    GST_ON_IMPORTS = "GST on Imports"

    @classmethod
    def from_string(cls, value: str) -> Optional['TaxName']:
        try:
            return cls(value)
        except (ValueError, AttributeError):
            return None


def predict_tax_name(
        tenant_id: str,
        start_date: str,
        end_date: str,
        logger: logging.Logger
) -> Tuple[JournalLinePredictionResponse, str]:
    gst_anomalies = []
    model_version = None
    table2df = tenant_data_collection(tenant_id=tenant_id,
                                      start_date=start_date,
                                      end_date=end_date,
                                      logger=logger)
    for table_name, df in table2df.items():
        if len(df) == 0:
            continue
        df_records = [JournalLineRecord(**{**data, "table": f"{table_name}"})
                      for data in df.to_dict('records')]
        table_request = JournalLinePredictionRequest(tenant_id=tenant_id,
                                                     table=table_name,
                                                     data=df_records)
        try:
            table_response = gst_validation(request=table_request)
        except Exception as e:
            logger.error(f"Error processing {table_name} records with model: {str(e)}")
            continue

        if not model_version:
            model_version = table_response.model_hyperparameters.version

        # filtering the response
        filtered_table_response_results = filter_gst_anomalies(response=table_response)

        # adding to the main list
        gst_anomalies.extend(filtered_table_response_results)
    response = JournalLinePredictionResponse(
        tenant_id=tenant_id,
        results=gst_anomalies
    )
    return response, model_version


async def main():
    # ===================== Configurations =====================
    TENANT_ID = "77b67978-ae76-4480-ac96-066ea269541e"
    TABLE_NAME = "cashrec_cashpaid"
    SERVICE = "gst_validation"
    RAW_DATA_TIMESTAMP = "2025-08-07-03-58-27"
    NUM_TEST_SAMPLES = 10

    # ===================== Loading Raw Data =====================
    raw_data_dir = f"/Users/amirhossein/Projects/bookiewand/ai/data/{SERVICE}/raw/{RAW_DATA_TIMESTAMP}"
    table_data_dir = os.path.join(raw_data_dir, TABLE_NAME)
    tenant_table_data_path = os.path.join(table_data_dir, f"{TENANT_ID}.csv")
    raw_df = pd.read_csv(tenant_table_data_path)
    raw_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ===================== Constructing Service Request =====================
    records = [JournalLineRecord(**{**data, "table": f"{TABLE_NAME}"})
               for data in raw_df.to_dict('records')[:NUM_TEST_SAMPLES]]
    service_request = JournalLinePredictionRequest(tenant_id=TENANT_ID,
                                                   table=TABLE_NAME,
                                                   data=records)

    # ===================== Debugging the APIs =====================
    # /model_version
    # try:
    #     res = get_model_version()
    # except Exception as e:
    #     raise e

    # /refresh_model
    # try:
    #     res = refresh_model()
    # except Exception as e:
    #     raise e

    # /gst_validation
    # try:
    #     res = gst_validation(request=service_request)
    # except Exception as e:
    #     raise e
    res = await llm_gst_validation(request=service_request)

    logger = get_logger()
    res = predict_tax_name(tenant_id="77b67978-ae76-4480-ac96-066ea269541e",
                           start_date="2025-03-01",
                           end_date="2025-04-30",
                           logger=logger)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())