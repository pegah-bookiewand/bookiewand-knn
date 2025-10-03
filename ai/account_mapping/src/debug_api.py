import os
import logging
import random
from typing import Tuple, List

import pandas as pd

from ai.utils.data_types import JournalLineRecord, JournalLinePredictionRequest, JournalLinePredictionResponse, \
    JournalLinePredictionResult
from ai.utils.data_collection import tenant_data_collection
from common.logs import get_logger
from ai.account_mapping.src.api import (
    get_model_version,
    refresh_model,
    account_mapping,
    llm_account_mapping
)


def filter_account_mapping_anomalies(
        response: JournalLinePredictionResponse
) -> List[JournalLinePredictionResult]:
    filtered_results = []
    for result in response.results:
        # Convert target_prediction to integer for comparison with account_code
        target_prediction_num = int(result.target_prediction) if result.target_prediction else None
        if result.record.root.account_code != target_prediction_num:
            filtered_results.append(result)
    return filtered_results


def predict_account_mapping(
        tenant_id: str,
        start_date: str,
        end_date: str,
        logger: logging.Logger,
) -> Tuple[JournalLinePredictionResponse, str]:
    account_mapping_anomalies = []
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
            table_response = account_mapping(request=table_request)
        except Exception as e:
            logger.error(f"Error processing {table_name} records with model: {str(e)}")
            raise e

        if not model_version:
            model_version = table_response.model_hyperparameters.version

        # filtering the response
        filtered_table_response_results = filter_account_mapping_anomalies(response=table_response)

        # adding to the main list
        account_mapping_anomalies.extend(filtered_table_response_results)
    response = JournalLinePredictionResponse(
        tenant_id=tenant_id,
        results=account_mapping_anomalies
    )
    return response, model_version


async def main():
    # ===================== Configurations =====================
    TENANT_ID = "77b67978-ae76-4480-ac96-066ea269541e"
    # TENANT_ID = "bb18b49f-d918-4550-bdd0-2e329b6614d7"
    TABLE_NAME = "cashrec_cashpaid"
    # TABLE_NAME = "accrec_accpay"
    SERVICE = "account_mapping"
    RAW_DATA_TIMESTAMP = "2025-08-06-09-37-47"
    NUM_TEST_SAMPLES = 5

    # ===================== Loading Raw Data =====================
    raw_data_dir = f"/Users/amirhossein/Projects/bookiewand/ai/data/{SERVICE}/raw/{RAW_DATA_TIMESTAMP}"
    table_data_dir = os.path.join(raw_data_dir, TABLE_NAME)
    tenant_table_data_path = os.path.join(table_data_dir, f"{TENANT_ID}.csv")
    raw_df = pd.read_csv(tenant_table_data_path)
    raw_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # =====================================================
    # -------- Issue 1 --------
    # TENANT_ID = "77b67978-ae76-4480-ac96-066ea269541e"
    # TABLE_NAME = "cashrec_cashpaid"
    # SERVICE = "account_mapping"
    # df[(df['account_code'] == '21997.0') & (df['contact_name'] == 'Central Highlands Water') & (
    #             df['description'] == 'Water & Rates - Wattletree Drv')]
    # LLM Output
    # LLMJournalAttributeInferenceOutput(
    #     target_variable='Loan - Tim & Sara McLeod',
    #     reasoning="The query row's description ('Water & Rates - Wattletree Drv') and contact ('Central Highlands Water') are an exact match to several similar historical records, all of which are consistently classified as 'Loan - Tim & Sara McLeod' when the bank account is 'Saneil Pty Ltd.' This strong pattern is reinforced by both the description and contact name, making it the most appropriate classification from the available options.",
    #     confidence_score=0.98
    # )
    debug_records = [
        {
            'account_code': '21997.0',
            'account_name': 'Loan - Tim & Sara McLeod',
            'account_type': 'LIABILITY',
            'bank_account_code': 10103.0,
            'bank_account_name': 'Saneil Pty Ltd',
            'bank_transaction_unique_id': 48034,
            'contact_name': 'Central Highlands Water',
            'description': 'Water & Rates - Wattletree Drv',
            'id': 48513,
            'journal_date': pd.Timestamp('2025-06-30 00:00:00'),
            'line_amount': 371.93,
            'line_item_id': 'ee92cc1f-129b-400e-893f-1c7756dc8d54',
            'source_id': '823e0b4f-b751-4ea9-8a99-744db85ef63e',
            'source_type': 'CASHPAID',
            'sub_total': 371.93,
            'tax_amount': 0.0,
            'tax_type': 'BASEXCLUDED',
            'tenant_id': '77b67978-ae76-4480-ac96-066ea269541e',
            'total': 371.93,
            'total_tax': 0.0,
            'unit_amount': 371.93
        }
    ]
    # --------  Issue 2 --------
    # TENANT_ID = "bb18b49f-d918-4550-bdd0-2e329b6614d7"
    # TABLE_NAME = "accrec_accpay"
    # raw_df[(raw_df['account_name'] == 'Tyres') & (raw_df['contact_name'] == 'Krysdion Dunlop')].to_dict('records')
    # debug_records = [{'account_code': 61820.0, 'account_name': 'Tyres', 'account_type': 'DIRECTCOSTS',
    #                   'contact_name': 'Krysdion Dunlop', 'description': 'Tyres bought on business account for Krys',
    #                   'id': 55646,
    #                   'invoice_number': 'INV-3438', 'invoice_unique_id': 27510, 'item_code': np.nan,
    #                   'journal_date': '2025-06-04',
    #                   'line_amount': 775.0, 'line_item_id': '05c9055f-0cac-42b7-8862-068f1e76f311', 'quantity': 1.0,
    #                   'reference': 'Employee Loan - Tyres', 'source_id': 'bd3e80b0-0222-49fd-81ad-9b8e56f289b4',
    #                   'source_type': 'ACCREC', 'sub_total': 704.55, 'tax_amount': 70.45, 'tax_type': 'INPUT',
    #                   'tenant_id': 'bb18b49f-d918-4550-bdd0-2e329b6614d7', 'unit_amount': 775.0}]
    # LLMJournalAttributeInferenceOutput(target_variable='Tyres',
    #                                    reasoning="The query row's description explicitly states 'Tyres bought on business account for Krys,' and the most similar historical record with the exact same description and contact name ('Krysdion Dunlop') was classified as 'Tyres.' No other account name in the available list is as directly relevant, and there is a clear historical precedent for this classification. Therefore, 'Tyres' is the most appropriate account name for this transaction.",
    #                                    confidence_score=1.0)
    pass

    # records = [JournalLineRecord(**{**data, "table": f"{TABLE_NAME}"})
    #            for data in debug_records]
    # =====================================================
    debug_record_0 = JournalLineRecord(**{**debug_records[0], "table": f"{TABLE_NAME}"})
    # ===================== Constructing Service Request =====================
    records = [JournalLineRecord(**{**data, "table": f"{TABLE_NAME}"})
               for data in raw_df.to_dict('records')[:NUM_TEST_SAMPLES]]
    # ========
    # Check if debug_records[0] exists in records
    found_index = None
    for i, record in enumerate(records):
        if record == debug_record_0:
            found_index = i
            break

    if found_index is not None:
        print(f"debug_records[0] found at index: {found_index}")
    else:
        records.append(debug_record_0)
        print(f"debug_records[0] not found, added at index: {len(records) - 1}")

    random.shuffle(records)
    # ========
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

    # /account_mapping
    # try:
    #     res = account_mapping(request=service_request)
    # except Exception as e:
    #     raise e

    # /llm_account_mapping
    res = await llm_account_mapping(request=service_request)

    logger = get_logger()
    res = predict_account_mapping(tenant_id="77b67978-ae76-4480-ac96-066ea269541e",
                                  start_date="2023-01-01",
                                  end_date="2025-04-30",
                                  logger=logger)
    pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
