import logging
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from scipy.sparse import csr_matrix, hstack

from ai.ml.knn.preprocessing_utils import (
    fit_transform_numerical,
    fit_transform_categorical,
    general_preprocessing
)


PREPROCESSING_OUTPUT = Dict[
    str, Union[
        str,
        List[str],
        pd.DataFrame,
        csr_matrix,
        StandardScaler,
        OneHotEncoder,
        pd.Series,
        Dict[str, int]
    ]
]


def preprocess_numerical_features(
        df: pd.DataFrame,
        numerical_features: List[str]
) -> Tuple[csr_matrix, Union[pd.Series, float], StandardScaler]:
    # Check if numerical_features list is empty or dataframe is empty
    if not numerical_features or df.empty:
        return csr_matrix((df.shape[0], 0)), pd.Series(), None
        
    # Handling missing values in numerical columns
    df.loc[:, numerical_features] = df.loc[:, numerical_features].replace({None: np.nan})
    NAN_IMPUTER_VALUE = 0.0
    df.loc[:, numerical_features] = df.loc[:, numerical_features].fillna(NAN_IMPUTER_VALUE)

    # Standardizing numerical features
    X_numerical, numerical_scaler = fit_transform_numerical(df=df,
                                                            numerical_columns=numerical_features)
    X_numerical = csr_matrix(X_numerical)
    return X_numerical, NAN_IMPUTER_VALUE, numerical_scaler


def preprocess_categorical_features(
        df: pd.DataFrame,
        categorical_features: List[str]
) -> Tuple[csr_matrix, OneHotEncoder]:
    # Handling missing values and converting categorical columns
    df[categorical_features] = df[categorical_features].replace({None: np.nan})

    # Encoding categorical features
    X_categorical, categorical_encoder = fit_transform_categorical(df=df,
                                                                   categorical_columns=categorical_features)
    X_categorical = csr_matrix(X_categorical)
    return X_categorical, categorical_encoder


def preprocessing_step(
        raw_df: pd.DataFrame,
        target_variable: str,
        categorical_features: List[str],
        numerical_features: List[str],
        filter_low_frequency_class_counts: bool,
        random_seed: int,
        logger: logging.Logger
) -> PREPROCESSING_OUTPUT:
    """
    Preprocess a raw dataframe by handling missing values, filtering the low frequency labels,
    and scaling/encoding numerical and categorical features.

    Parameters
    ----------
    raw_df : pd.DataFrame
        The input dataframe containing the raw data.
    target_variable : str
        The column name of the target variable.
    categorical_features : List[str]
        List of names for categorical columns that require encoding.
    numerical_features : List[str]
        List of names for numerical columns that require scaling.
    filter_low_frequency_class_counts : bool
        Flag indicating whether to filter out low frequency labels in the target variable.
    random_seed: int
    logger : logging.Logger
        Logger instance for logging informative messages and error tracking.

    Returns
    -------
    Dict[str, Union[str, List[str], pd.DataFrame, StandardScaler, OneHotEncoder, pd.Series, Dict[str, int]]]
        Dictionary containing:
            - "preprocessing_state": (str) Status of preprocessing (e.g., 'FINISHED', 'EMPTY_DATAFRAME', etc.).
            - "ignore_columns": (List[str]) List of columns that were ignored during preprocessing.
            - "numerical_columns": (List[str]) Updated list of numerical columns.
            - "categorical_columns": (List[str]) Updated list of categorical columns.
            - "df": (pd.DataFrame) The preprocessed dataframe.
            - "X": (scipy.sparse matrix) The combined feature matrix from numerical and categorical features.
            - "y": (pd.Series) The target variable series.
            - "numerical_medians": (pd.Series) Medians for each numerical column used to fill missing values.
            - "numerical_scaler": (StandardScaler) Fitted scaler for numerical columns.
            - "categorical_encoder": (OneHotEncoder) Fitted encoder for categorical columns.
            - "class_counts": (Dict[str, int]) Original counts of each class in the target variable.
            - "removed_classes": (List[str]) Classes that were removed due to low frequency when filtering is enabled.
    """
    logger.info("Preprocessing...")
    preprocessing_state = None

    # Log initial state of target variable
    logger.info(f"Initial unique values in {target_variable}: {raw_df[target_variable].unique()}")
    logger.info(f"Initial null count in {target_variable}: {raw_df[target_variable].isna().sum()}")
    logger.info(f"Initial None count in {target_variable}: {raw_df[target_variable].isnull().sum()}")

    # Creating a copy of the raw dataframe for preprocessing
    df = raw_df.copy()
    logger.info("Dataframe copied for preprocessing.")

    # ======== Preprocessing the target variable ========    
    output = general_preprocessing(
        df=df,
        target_variable=target_variable,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        filter_low_frequency_class_counts=filter_low_frequency_class_counts,
        random_seed=random_seed,
        logger=logger
    )
    df = output["df"]
    numerical_features = output["numerical_features"]
    categorical_features = output["categorical_features"]
    constant_columns = output["constant_columns"]
    removed_classes = output["removed_classes"]
    class_counts = output["class_counts"]
    
    # Check for empty dataframe early
    if df.empty:
        logger.warning("Dataframe is empty after preprocessing target variable and filtering.")
        preprocessing_state = "EMPTY_DATAFRAME"
        
        # Return early with empty structures
        return {
            "preprocessing_state": preprocessing_state,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "raw_df": raw_df,
            "train_df": df,
            "X": csr_matrix((0, len(numerical_features) + len(categorical_features))),
            "y": pd.Series(),
            "numerical_medians": pd.Series(),
            "numerical_scaler": None,
            "categorical_encoder": None,
            "class_counts": class_counts.to_dict(),
            "removed_classes": removed_classes,
            "constant_columns": constant_columns
        }
    
    if target_variable == constant_columns:
        preprocessing_state = "CONSTANT_TARGET_VARIABLE"

    # ======== Preprocessing the numerical features ========
    if numerical_features:
        X_numerical, nan_imputer_value, numerical_scaler = preprocess_numerical_features(
            df=df,
            numerical_features=numerical_features
        )
    else:
        nan_imputer_value = None
        X_numerical = csr_matrix((df.shape[0], 0))
        numerical_scaler = None

    # ======== Preprocessing the categorical features ========
    if categorical_features:
        X_categorical, categorical_encoder = preprocess_categorical_features(
            df=df,
            categorical_features=categorical_features
        )
    else:
        X_categorical = csr_matrix((df.shape[0], 0))
        categorical_encoder = None

    X = hstack([X_numerical, X_categorical])
    y = df[target_variable]
    logger.info("Combined numerical and categorical features into feature matrix X.")

    # Updating the preprocessing state
    if X.shape[0] == 0:
        preprocessing_state = "EMPTY_FEATURE_MATRIX"
        logger.warning("Feature matrix X is empty after preprocessing.")
    else:
        if preprocessing_state is None:
            preprocessing_state = "FINISHED"
        logger.info("Preprocessing finished successfully.")

    processed_data_dict = {
        "preprocessing_state": preprocessing_state,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "raw_df": raw_df,
        "train_df": df,
        "X": X,
        "y": y,
        "nan_imputer_value": nan_imputer_value,
        "numerical_scaler": numerical_scaler,
        "categorical_encoder": categorical_encoder,
        "class_counts": class_counts.to_dict(),
        "removed_classes": removed_classes,
        "constant_columns": constant_columns
    }

    logger.info("Processed data dictionary created with preprocessing state: %s", preprocessing_state)
    return processed_data_dict
