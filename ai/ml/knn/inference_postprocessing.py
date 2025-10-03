import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix

from ai.utils.data_types import (
    JournalLineRecord,
    JournalLinePredictionSimilarExamples,
    JournalLinePredictionResult,
    ModelHyperparameters,
    JournalLinePredictionResponse
)


def convert_float_string_to_int_string(value):
    if isinstance(value, str):
        try:
            float_val = float(value)
            if float_val.is_integer():
                return str(int(float_val))
        except (ValueError, TypeError):
            pass
    return str(value) if value is not None else value


def postprocessing_step(
        tenant_id: str,
        table_name: str,
        predictions: np.ndarray,
        X: csr_matrix,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        train_class_counts: Dict[str, int],
        train_removed_classes: List[str],
        knn_model: KNeighborsClassifier,
        model_hyperparameters: ModelHyperparameters,
        logger: logging.Logger
) -> JournalLinePredictionResponse:

    logger.info(f"Generating explanations for {len(predictions)} predictions.")
    target_variable = model_hyperparameters.target_variable
    distances, indices = knn_model.kneighbors(X.toarray())

    # preparing the results
    results = []
    for i, (pred, dists, idxs) in enumerate(zip(predictions, distances, indices)):
        pred = convert_float_string_to_int_string(pred)
        record = {k: v for k, v in df.iloc[i].to_dict().items()}
        record = JournalLineRecord(**{**record, "table": table_name})

        # Finding neighbors data in raw df
        merge_columns = [col for col in train_df.columns if 'id' in col.lower() and col.lower() != 'tenant_id']
        neighbors_train_df = train_df.iloc[idxs]
        neighbors_train_raw_df = raw_df.merge(neighbors_train_df[merge_columns], on=merge_columns, how='inner')
        neighbors_data = neighbors_train_raw_df.to_dict('records')

        # Build the explanation for this sample
        explanations = []
        for dist, idx, train_record in zip(dists, idxs, neighbors_data):
            train_record = JournalLineRecord(**{**train_record, "tenant_id": tenant_id, "table": table_name})
            explanation = JournalLinePredictionSimilarExamples(distance=dist,
                                                               train_index=int(idx),
                                                               record=train_record)
            explanations.append(explanation)

        user_class = df.iloc[i][target_variable]
        user_class = convert_float_string_to_int_string(user_class)
        if user_class in train_removed_classes:
            logger.info(f"Sample {i} with REMOVED class during training {user_class}: "
                        f"Setting user_class_frequency=0.")
            user_class_frequency = 0
        elif user_class not in train_removed_classes and user_class not in train_class_counts:
            # novel class (not seen during training)
            logger.info(f"Sample {i} with NOVEL class unseen during training {user_class}: "
                        f"Setting user_class_frequency=0.")
            user_class_frequency = 0
        else:
            user_class_frequency = (train_class_counts[user_class]/sum(train_class_counts.values()))
        user_class = convert_float_string_to_int_string(user_class)

        sample_results = JournalLinePredictionResult(target_prediction=pred,
                                                     record=record,
                                                     user_class=user_class,
                                                     user_class_frequency=user_class_frequency,
                                                     explanations=explanations)
        results.append(sample_results)
    response = JournalLinePredictionResponse(tenant_id=tenant_id,
                                             results=results,
                                             model_hyperparameters=model_hyperparameters)
    logger.info("Postprocessing complete.")
    return response
