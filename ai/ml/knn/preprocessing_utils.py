import logging
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fit_transform_numerical(
        df: pd.DataFrame,
        numerical_columns: List[str]
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on numerical columns and transform the data.

    This function fits a StandardScaler using the specified numerical columns from the
    input DataFrame. It then transforms these columns into scaled values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    numerical_columns : list of str
        List of numerical column names to be scaled.

    Returns
    -------
    x_numerical : np.ndarray
        Array of scaled numerical values.
    scaler : StandardScaler
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    numerical_df = df[numerical_columns]
    x_numerical = scaler.fit_transform(numerical_df)
    return x_numerical, scaler


def transform_numerical(
        df: pd.DataFrame,
        numerical_columns: List[str],
        scaler: StandardScaler
) -> np.ndarray:
    """
    Transform numerical columns using an existing StandardScaler.

    This function applies the transformation provided by the fitted StandardScaler
    on the specified numerical columns in the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to transform.
    numerical_columns : list of str
        List of numerical column names to be transformed.
    scaler : StandardScaler
        Fitted StandardScaler instance used for scaling.

    Returns
    -------
    x_numerical : np.ndarray
        Array of transformed numerical values.
    """
    numerical_df = df[numerical_columns]
    x_numerical = scaler.transform(numerical_df)
    return x_numerical


def fit_transform_categorical(
        df: pd.DataFrame,
        categorical_columns: List[str]
) -> Tuple[np.ndarray, OneHotEncoder]:
    """
    Fit a OneHotEncoder on categorical columns and transform the data.

    This function fits a OneHotEncoder on the specified categorical columns of the
    input DataFrame and transforms these columns into one-hot encoded features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    categorical_columns : list of str
        List of categorical column names to be encoded.

    Returns
    -------
    x_categorical : np.ndarray
        Array of one-hot encoded categorical features.
    encoder : OneHotEncoder
        Fitted OneHotEncoder instance.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_categorical = encoder.fit_transform(df[categorical_columns])
    return x_categorical, encoder


def transform_categorical(
        df: pd.DataFrame,
        categorical_columns: List[str],
        encoder: OneHotEncoder
) -> np.ndarray:
    """
    Transform categorical columns using an existing OneHotEncoder.

    This function applies the fitted OneHotEncoder to the specified categorical columns
    in the input DataFrame, returning the one-hot encoded representation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to transform.
    categorical_columns : list of str
        List of categorical column names to be encoded.
    encoder : OneHotEncoder
        Fitted OneHotEncoder instance used for encoding.

    Returns
    -------
    x_categorical : np.ndarray
        Array of one-hot encoded categorical features.
    """
    return encoder.transform(df[categorical_columns])


def validate_model_config(
        raw_df: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
        columns: List[str],
        target_variable: str,
        date_column: str,
        table_name: str = None
) -> None:
    """
    Validate that the model configuration aligns with the input dataframe by checking the consistency
    between the defined columns for the setup and the columns available in the dataframe.

    Parameters
    ----------
    raw_df : pd.DataFrame
        The dataframe containing the data to be used for model configuration.
    categorical_features : List[str]
        A list of column names considered as categorical features.
    numerical_features : List[str]
        A list of column names considered as numerical features.
    columns : List[str]
        A list of supposed column names of raw_df.
    target_variable : str
        The name of the target variable column which the model will predict.
    date_column : str
        The name of the date column in the dataframe.
    table_name : str, optional
        The name of the table being validated, used for more detailed error messages.

    Returns
    -------
    None
        This function does not return a value but raises an AssertionError if any validation check fails.

    Raises
    ------
    AssertionError
        If the total number of columns in the dataframe is not equal to the sum of ignore, categorical,
        numerical columns plus one (for the target variable), or if any of the specified columns (target, date,
        ignore, categorical, numerical) do not exist in the dataframe.
    """
    table_info = f" for table '{table_name}'" if table_name else ""
    
    # Check if the number of columns in DataFrame matches the expected number
    assert len(raw_df.columns) == len(columns), \
        f"Number of columns in the data{table_info} is not equal to the defined columns of the setup, i.e., " \
        f"{len(raw_df.columns)} != {len(columns)}.\n" \
        f"DataFrame columns: {list(raw_df.columns)}\n" \
        f"Expected columns: {columns}"
    
    # Check if target variable exists in the DataFrame
    assert target_variable in raw_df.columns, \
        f"The dataframe{table_info} does not have the target variable column '{target_variable}'.\n" \
        f"Available columns: {list(raw_df.columns)}"
    
    # Check if date column exists in the DataFrame
    assert date_column in raw_df.columns, \
        f"The dataframe{table_info} does not have the date column '{date_column}'.\n" \
        f"Available columns: {list(raw_df.columns)}"
    
    # Check if all categorical features exist in the DataFrame
    missing_categorical = [col for col in categorical_features if col not in raw_df.columns]
    assert not missing_categorical, \
        f"The dataframe{table_info} is missing the following categorical features: {missing_categorical}.\n" \
        f"Available columns: {list(raw_df.columns)}"
    
    # Check if all numerical features exist in the DataFrame
    missing_numerical = [col for col in numerical_features if col not in raw_df.columns]
    assert not missing_numerical, \
        f"The dataframe{table_info} is missing the following numerical features: {missing_numerical}.\n" \
        f"Available columns: {list(raw_df.columns)}"


def preprocess_target_variable(
        df: pd.DataFrame,
        target_variable: str,
        logger: logging.Logger
) -> pd.DataFrame:
    """
    Preprocess the target variable by converting it to a string, handling missing values,
    and filling missing values with a default label ('NOT_APPLICABLE').

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the target variable column to preprocess.
    target_variable : str
        Name of the column that represents the target variable.
    logger : logging.Logger
        Logger instance for logging messages about the processing steps.

    Returns
    -------
    pd.DataFrame
        The dataframe with the target variable column preprocessed.
    """
    # Convert target variable to string and handle all types of missing values
    df[target_variable] = df[target_variable].astype(str)
    df[target_variable] = df[target_variable].replace(
        {
            None: np.nan,
            'None': np.nan,
            'nan': np.nan,
            'NaN': np.nan
        }
    )
    nan_count = df[target_variable].isna().sum()
    logger.info("Converted target variable to string and replaced specified missing values.")
    logger.info(f"Found {nan_count} missing values in target variable '{target_variable}'.")

    # Filling the nan target variable values with NOT_APPLICABLE label
    df[target_variable] = df[target_variable].fillna('NOT_APPLICABLE')
    logger.info(f"Filled missing values in target variable '{target_variable}' with 'NOT_APPLICABLE'.")
    logger.info(f"Final unique values in {target_variable}: {df[target_variable].unique()}")
    logger.info("Data shape after handling missing target values: %s", df.shape)
    return df


def filter_classes(
        df: pd.DataFrame,
        target_variable: str,
        mode: str,
        logger: logging.Logger,
        threshold_constant: int = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Filters out low-frequency classes in the target variable of a dataframe based on the specified mode.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data, including the target variable.
    target_variable : str
        The name of the column in the dataframe representing the target variable.
    mode : str
        Mode of filtering low-frequency classes. Options:
        - "less_than_1pct_of_majority": Removes classes with counts less than 1% of the majority class.
        - "constant": Removes classes with fewer samples than a user-defined constant threshold.
    logger : logging.Logger
        Logger instance for logging filtering activities and errors.
    threshold_constant : int, optional
        Custom threshold for minimum class count when mode is "constant".
        Required if `mode` is "constant". Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        - pd.DataFrame: The modified dataframe after filtering low-frequency classes.
        - pd.Series: The class counts of the target variable before filtering.
        - List[str]: The names of the classes that were removed during filtering.
    """
    class_counts = df[target_variable].value_counts()
    removed_classes = []

    if mode == "less_than_1pct_of_majority":
        # threshold is 1% of the maximum class count
        threshold = int(class_counts.max() * 0.01)
        valid_classes = class_counts[class_counts >= threshold].index
        removed_classes = list(class_counts[class_counts < threshold].index)
        df = df[df[target_variable].isin(valid_classes)]
        logger.info("Filtered out low frequency labels in target variable based on threshold %s.", threshold)

    elif mode == "constant":
        if threshold_constant is None:
            raise ValueError("You must provide 'threshold_constant' for 'constant' mode.")
        valid_classes = class_counts[class_counts >= threshold_constant].index
        removed_classes = list(class_counts[class_counts < threshold_constant].index)
        df = df[df[target_variable].isin(valid_classes)]
        logger.info("Filtered out classes with fewer than %s samples.", threshold_constant)

    return df, class_counts, removed_classes


def filter_constant_columns(
        df: pd.DataFrame,
        target_variable: str,
        categorical_features: List[str],
        numerical_features: List[str],
        logger: logging.Logger
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Filters out columns in a DataFrame that have only a single unique value
    and updates the lists of categorical and numerical features accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to process.
    target_variable: str
    categorical_features : List[str]
        List of column names corresponding to categorical features.
    numerical_features : List[str]
        List of column names corresponding to numerical features.
    logger : logging.Logger
        Logger instance for logging information about removed columns.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str], List[str]]
        - Filtered DataFrame with constant columns removed.
        - List of removed constant column names.
        - Updated list of categorical features with constant columns removed.
        - Updated list of numerical features with constant columns removed.
    """
    # Removing columns with only a single unique value
    constant_columns = [col for col in df.columns if len(df[col].unique()) == 1 and col != target_variable]
    if constant_columns:
        logger.info("Removing constant columns: %s", constant_columns)
    df = df.drop(columns=constant_columns)
    for col in constant_columns:
        if col in categorical_features:
            categorical_features.remove(col)

        if col in numerical_features:
            numerical_features.remove(col)

    return df, constant_columns, categorical_features, numerical_features


def general_preprocessing(
        df: pd.DataFrame, 
        target_variable: str,
        numerical_features: List[str],
        categorical_features: List[str],
        filter_low_frequency_class_counts: bool,
        random_seed: int,
        logger: logging.Logger
) -> Dict:
    df = preprocess_target_variable(df=df, target_variable=target_variable, logger=logger)

    if filter_low_frequency_class_counts:
        df, class_counts, removed_classes = filter_classes(df=df,
                                                           target_variable=target_variable,
                                                           mode='constant',
                                                           threshold_constant=5,
                                                           logger=logger)
    else:
        class_counts = df[target_variable].value_counts()
        removed_classes = []
        logger.info("Skipping low frequency label filtering for target variable.")

    # Removing constant columns with only a single unique value
    df, constant_columns, categorical_features, numerical_features = filter_constant_columns(
        df=df,
        target_variable=target_variable,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        logger=logger
    )
    # shuffling the dataset
    df = df.sample(frac=1, random_state=random_seed)
    df.reset_index(inplace=True, drop=True)
    output = {
        "df": df,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "constant_columns": constant_columns,
        "removed_classes": removed_classes,
        "class_counts": class_counts
    }
    return output
