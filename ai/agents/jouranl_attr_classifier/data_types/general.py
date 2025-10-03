from typing import List, Dict, Union, Optional

from pydantic import BaseModel

from ai.utils.data_types import JournalLineRecord


class LLMJournalAttributeInferenceOutput(BaseModel):
    target_variable: str
    reasoning: str
    confidence_score: float


class LLMJournalLinePredictionSimilarExamples(BaseModel):
    feature: str
    train_index: int
    distance: float
    record: JournalLineRecord


class LLMJournalLinePredictionResult(BaseModel):
    record: JournalLineRecord
    target_prediction: str
    user_class: str
    confidence_score: float
    reasoning: str
    error_explanation: Optional[str] = None
    explanations: List[LLMJournalLinePredictionSimilarExamples]


class Hyperparameters(BaseModel):
    run_timestamp: str
    table_name: str
    vectorizer: str
    raw_data_path: str
    checkpoints_dir: str
    features: List[str]
    date_column: str
    target_variable: str
    n_neighbors: int
    num_samples: int
    llm_azure_endpoint: str
    llm_api_version: str
    llm_model_name: str
    llm_temperature: float
    llm_top_p: float


class LLMJournalLinePredictionResponse(BaseModel):
    tenant_id: str
    results: List[LLMJournalLinePredictionResult]
    hyperparameters: Hyperparameters
