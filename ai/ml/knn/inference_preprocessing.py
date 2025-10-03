import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from scipy.sparse import csr_matrix, hstack

from ai.ml.knn.preprocessing_utils import (
    transform_numerical,
    transform_categorical,
    preprocess_target_variable
)


def preprocess_numerical_features(
        df: pd.DataFrame,
        nan_imputer_value: float,
        numerical_scaler: StandardScaler,
        numerical_features: List[str],
) -> csr_matrix:
    # ================= Train Data =================
    # Filling None values with np.nan for train 
    df.loc[:, numerical_features] = df.loc[:, numerical_features].replace({None: np.nan})
    # Filling the nan values with numerical medians
    df.loc[:, numerical_features] = df.loc[:, numerical_features].fillna(nan_imputer_value)
    # standardizing the data
    X_numerical = transform_numerical(df=df,
                                      numerical_columns=numerical_features,
                                      scaler=numerical_scaler)
    # sparsifying the data
    X_numerical = csr_matrix(X_numerical)

    return X_numerical


def preprocess_categorical_features(
        df: pd.DataFrame,
        categorical_encoder: OneHotEncoder,
        categorical_features: List[str]
) -> csr_matrix:
    df.loc[:, categorical_features] = df.loc[:, categorical_features].replace({None: np.nan})
    X_categorical = transform_categorical(df=df,
                                          categorical_columns=categorical_features,
                                          encoder=categorical_encoder)
    X_categorical = csr_matrix(X_categorical)
    return X_categorical


def preprocessing_step(
        df: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
        target_variable: str,
        nan_imputer_value: float,
        numerical_scaler: StandardScaler,
        categorical_encoder: OneHotEncoder,
        logger: logging.Logger
) -> csr_matrix:
    logger.debug("Starting data preprocessing.")
    logger.debug(f"Data shape before preprocessing: {df.shape}")

    df = preprocess_target_variable(df=df, target_variable=target_variable, logger=logger)

    # preprocessing numerical features
    if numerical_features:
        X_numerical = preprocess_numerical_features(df=df,
                                                    nan_imputer_value=nan_imputer_value,
                                                    numerical_scaler=numerical_scaler,
                                                    numerical_features=numerical_features)
    else:
        X_numerical = csr_matrix((df.shape[0], 0))

    # preprocessing categorical features
    if categorical_features:
        X_categorical = preprocess_categorical_features(df=df,
                                                        categorical_encoder=categorical_encoder,
                                                        categorical_features=categorical_features)
    else:
        X_categorical = csr_matrix((df.shape[0], 0))
    X = hstack([X_numerical, X_categorical])
    logger.info(f"Shape of feature matrix X: {X.shape}")
    return X
