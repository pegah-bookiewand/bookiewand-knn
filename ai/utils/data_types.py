import math

from typing import List, Dict, Optional, Union, Any, Literal, Annotated, ClassVar
import pandas as pd
from pydantic import BaseModel, Field, model_validator, RootModel, field_validator


class CleanBaseModel(BaseModel):
    categorical_fields: ClassVar[list[str]] = []
    numerical_fields: ClassVar[list[str]] = []
    bool_fields: ClassVar[list[str]] = []

    @model_validator(mode="before")
    def clean_values(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        for field_name, raw_value in data.items():
            # Normalize string representations of missing values.
            if isinstance(raw_value, str) and raw_value.lower() in {"nan", "none"}:
                data[field_name] = None
                continue
            if isinstance(raw_value, float) and math.isnan(raw_value):
                data[field_name] = None
                continue

            # For categorical fields: cast any non-None value to string.
            if field_name in cls.categorical_fields:
                if raw_value is not None:
                    data[field_name] = str(raw_value)

            # For numerical fields: cast any non-None value to float.
            elif field_name in cls.numerical_fields:
                if raw_value is not None:
                    try:
                        data[field_name] = float(raw_value)
                    except (ValueError, TypeError):
                        data[field_name] = None

            # For boolean fields: convert string 'true'/'false' to bool.
            elif field_name in cls.bool_fields:
                if isinstance(raw_value, str):
                    low = raw_value.lower()
                    if low == "true":
                        data[field_name] = True
                    elif low == "false":
                        data[field_name] = False
                    elif low == "none":
                        data[field_name] = None
        return data


class SharedJournalLineColumns(CleanBaseModel):
    tenant_id: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    description: Optional[str] = None
    tax_type: Optional[str] = None
    account_code: Optional[str] = None
    account_name: Optional[str] = None
    account_type: Optional[str] = None
    line_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    total_tax: Optional[float] = None
    id: Optional[int] = None
    journal_date: Optional[str] = None
    reference: Optional[str] = None
    # New fields from train_config_test.yaml
    quantity: Optional[float] = None
    unit_amount: Optional[float] = None

    categorical_fields: ClassVar[list[str]] = [
        "tenant_id", "source_type", "source_id", "tax_type", "account_code", "account_name", 
        "account_type", "description", "journal_date"
    ]
    numerical_fields: ClassVar[list[str]] = [
        "line_amount", "tax_amount", "total_tax", "id", "quantity", "unit_amount"
    ]


class AccrecAccpayPayments(SharedJournalLineColumns):
    table: Literal['accrec_accpay_payments'] = 'accrec_accpay_payments'

    # Categorical columns
    payment_type: Optional[str] = None
    payment_status: Optional[str] = None
    payment_account_code: Optional[str] = None
    payment_invoice_number: Optional[str] = None
    payment_invoice_type: Optional[str] = None
    payment_contact_name: Optional[str] = None
    line_item_description: Optional[str] = None
    line_item_account_code: Optional[float] = None

    # Additional column
    payment_id: Optional[str] = None
    line_item_id: Optional[str] = None

    # Numerical columns
    payment_amount: Optional[float] = None
    payment_bank_amount: Optional[float] = None
    payment_total_line_amount: Optional[float] = None
    payment_line_item_count: Optional[float] = None
    line_item_amount: Optional[float] = None
    line_item_tax_amount: Optional[float] = None
    line_item_unit_amount: Optional[float] = None
    line_item_quantity: Optional[float] = None

    # Target variable
    line_item_tax_type: Optional[str] = None

    categorical_fields: ClassVar[list[str]] = SharedJournalLineColumns.categorical_fields + [
        "account_code", "account_name",
        "payment_type", "payment_status", "payment_account_code",
        "payment_invoice_number", "payment_invoice_type", "payment_contact_name",
        "line_item_description", "line_item_tax_type", "payment_id", "line_item_id",
        "payment_line_item_count"
    ]
    numerical_fields: ClassVar[list[str]] = SharedJournalLineColumns.numerical_fields + [
        "payment_amount", "payment_bank_amount", "payment_total_line_amount",
        "payment_line_item_count", "line_item_amount", "line_item_tax_amount",
        "line_item_unit_amount", "line_item_quantity"
    ]


class AccRecCreditAccPayCredit(SharedJournalLineColumns):
    table: Literal['accreccredit_accpaycredit'] = 'accreccredit_accpaycredit'

    # Categorical columns from train_config_test.yaml
    line_item_id: Optional[str] = None
    credit_note_unique_id: Optional[str] = None
    credit_note_id: Optional[str] = None
    contact_name: Optional[str] = None
    credit_note_date: Optional[str] = None
    credit_note_type: Optional[str] = None
    credit_note_status: Optional[str] = None
    credit_note_number: Optional[str] = None
    credit_note_reference: Optional[str] = None

    # Numerical columns from train_config_test.yaml
    sub_total: Optional[float] = None
    total: Optional[float] = None

    categorical_fields: ClassVar[list[str]] = SharedJournalLineColumns.categorical_fields + [
        "line_item_id", "credit_note_unique_id", "credit_note_id", "contact_name",
        "credit_note_date", "credit_note_type", "credit_note_status", 
        "credit_note_number", "credit_note_reference"
    ]
    numerical_fields: ClassVar[list[str]] = SharedJournalLineColumns.numerical_fields + [
        "sub_total", "total"
    ]


class CashrecCashpaid(SharedJournalLineColumns):
    table: Literal['cashrec_cashpaid'] = 'cashrec_cashpaid'

    # Categorical columns from train_config_test.yaml
    line_item_id: Optional[str] = None
    bank_transaction_unique_id: Optional[str] = None
    bank_transaction_id: Optional[str] = None
    contact_name: Optional[str] = None
    transaction_date: Optional[str] = None
    transaction_type: Optional[str] = None
    transaction_status: Optional[str] = None
    bank_account_code: Optional[str] = None
    bank_account_name: Optional[str] = None

    # Numerical columns from train_config_test.yaml
    sub_total: Optional[float] = None
    total: Optional[float] = None

    categorical_fields: ClassVar[list[str]] = SharedJournalLineColumns.categorical_fields + [
        "line_item_id", "bank_transaction_unique_id", "bank_transaction_id", "contact_name",
        "transaction_date", "transaction_type", "transaction_status",
        "bank_account_code", "bank_account_name"
    ]
    numerical_fields: ClassVar[list[str]] = SharedJournalLineColumns.numerical_fields + [
        "sub_total", "total"
    ]


class AccrecAccpay(SharedJournalLineColumns):
    table: Literal['accrec_accpay'] = 'accrec_accpay'

    # Categorical columns from train_config_test.yaml
    line_item_id: Optional[str] = None
    invoice_unique_id: Optional[str] = None
    invoice_id: Optional[str] = None
    contact_name: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_type: Optional[str] = None
    invoice_status: Optional[str] = None
    invoice_number: Optional[str] = None
    item_code: Optional[str] = None
    sub_total: Optional[float] = None
    tax_name: Optional[str] = None

    categorical_fields: ClassVar[list[str]] = SharedJournalLineColumns.categorical_fields + [
        "line_item_id", "invoice_unique_id", "invoice_id", "contact_name", 
        "invoice_date", "invoice_type", "invoice_status", "invoice_number", "item_code"
    ]
    numerical_fields: ClassVar[list[str]] = SharedJournalLineColumns.numerical_fields + [
        "sub_total"
    ]


class ExpClaim(SharedJournalLineColumns):
    table: Literal['expclaim'] = 'expclaim'

    # Categorical columns from train_config_test.yaml
    line_item_id: Optional[str] = None
    receipt_unique_id: Optional[str] = None
    receipt_id: Optional[str] = None
    receipt_number: Optional[str] = None
    contact_name: Optional[str] = None
    receipt_date: Optional[str] = None
    receipt_status: Optional[str] = None
    receipt_reference: Optional[str] = None
    expense_claim_id: Optional[str] = None
    expense_claim_status: Optional[str] = None
    user_email: Optional[str] = None
    user_first_name: Optional[str] = None
    user_last_name: Optional[str] = None
    user_organisation_role: Optional[str] = None

    # Numerical columns from train_config_test.yaml
    receipt_total: Optional[float] = None
    receipt_sub_total: Optional[float] = None
    receipt_total_tax: Optional[float] = None
    expense_claim_total: Optional[float] = None
    amount_due: Optional[float] = None
    amount_paid: Optional[float] = None

    categorical_fields: ClassVar[list[str]] = SharedJournalLineColumns.categorical_fields + [
        "line_item_id", "receipt_unique_id", "receipt_id", "receipt_number", "contact_name",
        "receipt_date", "receipt_status", "receipt_reference", "expense_claim_id",
        "expense_claim_status", "user_email", "user_first_name", "user_last_name",
        "user_organisation_role"
    ]
    numerical_fields: ClassVar[list[str]] = SharedJournalLineColumns.numerical_fields + [
        "receipt_total", "receipt_sub_total", "receipt_total_tax",
        "expense_claim_total", "amount_due", "amount_paid"
    ]


class JournalLineRecord(
    RootModel[
        Annotated[
            Union[
                AccrecAccpayPayments,
                CashrecCashpaid,
                AccrecAccpay,
                ExpClaim,
                AccRecCreditAccPayCredit
            ],
            Field(discriminator='table')
        ]
    ]
):
    def dict(self, *args, **kwargs):
        return self.root.dict(*args, **kwargs)


# TODO change the name of the request to something clearer
class JournalLinePredictionRequest(BaseModel):
    tenant_id: str
    table: str

    data: List[JournalLineRecord]

    @field_validator("data")
    def check_single_type(cls, v: List[JournalLineRecord]) -> List[JournalLineRecord]:
        if not v:
            return v
        # Get the type of the first inner record using the 'root' attribute.
        first_type = type(v[0].root)
        for item in v:
            if type(item.root) is not first_type:
                raise ValueError("All items in 'data' must be of the same type.")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        # Convert each record to a dictionary using our custom dict() method.
        return pd.DataFrame([record.dict() for record in self.data])


class JournalLinePredictionSimilarExamples(BaseModel):
    distance: float
    train_index: int
    record: JournalLineRecord


class JournalLinePredictionResult(BaseModel):
    target_prediction: str
    record: JournalLineRecord
    user_class: str
    user_class_frequency: float
    confidence: Optional[float] = None
    error_explanation: Optional[str] = None

    explanations: Optional[List[JournalLinePredictionSimilarExamples]] = None


class ModelHyperparameters(BaseModel):
    model_name: str
    n_neighbors: Optional[int] = None
    weights: Optional[str] = None
    random_seed: Optional[int] = None
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[Union[List[str], bool]] = None
    target_variable: Optional[str] = None
    removed_classes: Optional[List[str]] = None
    constant_columns: Optional[List[str]] = None
    filter_low_frequency_class_counts: Optional[bool] = None
    class_counts: Optional[Dict[str, int]] = None
    input_dim: Optional[int] = None
    num_samples: Optional[int] = None
    version: Optional[str] = None
    run_timestamp: Optional[str] = None
    table_name: Optional[str] = None


class JournalLinePredictionResponse(BaseModel):
    tenant_id: str
    results: List[JournalLinePredictionResult]
    model_hyperparameters: Optional[ModelHyperparameters] = None
