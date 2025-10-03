from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from scipy.sparse import csr_matrix

from ai.ml.knn.preprocessing_utils import (
    fit_transform_numerical,
    transform_numerical,
    fit_transform_categorical,
    transform_categorical
)


pd.set_option('future.no_silent_downcasting', True)


def preprocess_numerical_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numerical_features: List[str],
) -> Tuple[csr_matrix, csr_matrix, StandardScaler]:
    # ================= Train Data =================
    # Filling None values with np.nan for train
    train_df.loc[:, numerical_features] = train_df.loc[:, numerical_features].replace({None: np.nan})
    # Computing the median of numerical columns for the train df
    train_fold_numerical_medians = test_df[numerical_features].median()
    # Filling the nan values of train fold with train median
    train_df.loc[:, numerical_features] = train_df.loc[:, numerical_features].fillna(train_fold_numerical_medians)
    # standardizing the train data
    X_train_numerical, numerical_scaler = fit_transform_numerical(
        df=train_df,
        numerical_columns=numerical_features
    )
    # sparsifying the data
    X_train_numerical = csr_matrix(X_train_numerical)

    # ================= Test Data =================
    # Filling None values with np.nan for test
    test_df.loc[:, numerical_features] = test_df.loc[:, numerical_features].replace({None: np.nan})
    # Filling the nan values of test with train median
    test_df.loc[:, numerical_features] = test_df.loc[:, numerical_features].fillna(train_fold_numerical_medians)
    # standardizing the test data with train stats
    X_test_numerical = transform_numerical(df=test_df,
                                           numerical_columns=numerical_features,
                                           scaler=numerical_scaler)
    # sparsifying the data
    X_test_numerical = csr_matrix(X_test_numerical)

    return X_train_numerical, X_test_numerical, numerical_scaler


def preprocess_categorical_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_features: List[str]
) -> tuple[csr_matrix, csr_matrix, OneHotEncoder]:
    # replacing None with nan
    train_df[categorical_features] = train_df[categorical_features].replace({None: np.nan})
    test_df[categorical_features] = test_df[categorical_features].replace({None: np.nan})

    # encoding the train data
    X_train_categorical, categorical_encoder = fit_transform_categorical(
        df=train_df,
        categorical_columns=categorical_features
    )

    # encoding the test data with train categorical encoder
    X_test_categorical = transform_categorical(df=test_df,
                                               categorical_columns=categorical_features,
                                               encoder=categorical_encoder)

    # sparsifying the data
    X_train_categorical = csr_matrix(X_train_categorical)
    X_test_categorical = csr_matrix(X_test_categorical)

    return X_train_categorical, X_test_categorical, categorical_encoder
