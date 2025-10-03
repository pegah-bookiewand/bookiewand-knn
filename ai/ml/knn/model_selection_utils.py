import time
import logging
import warnings
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

from ai.ml.knn.model_selection_preprocessing import (
    preprocess_numerical_features,
    preprocess_categorical_features
)


def compute_features_target_mi(
        df: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
        target_variable: str
) -> List[str]:
    """
    Compute mutual information (MI) scores between features and the target variable,
    and rank the features based on their MI scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the features and the target variable.
    categorical_features : List[str]
        List of column names specifying the categorical features.
    numerical_features : List[str]
        List of column names specifying the numerical features.
    target_variable : str
        Name of the target variable column.

    Returns
    -------
    List[str]
        Ordered list of feature names ranked by their mutual information scores,
        with the most informative feature last.
    """
    mi_df = df.copy()
    mi_df.loc[:, numerical_features] = mi_df.loc[:, numerical_features].replace({None: np.nan})
    numerical_medians = mi_df[numerical_features].median()
    mi_df.loc[:, numerical_features] = mi_df.loc[:, numerical_features].fillna(numerical_medians)

    mi_df[categorical_features] = mi_df[categorical_features].astype(str)
    mi_df.loc[:, categorical_features] = mi_df.loc[:, categorical_features].replace({None: np.nan}).fillna(
        "BLANK")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    mi_df[categorical_features] = encoder.fit_transform(mi_df[categorical_features])

    current_features = tuple(sorted(numerical_features + categorical_features))
    X = mi_df[list(current_features)]
    y = mi_df[target_variable]
    discrete_features = np.array([feature in categorical_features for feature in current_features])
    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)

    # Create a DataFrame to display the results
    mi_df_summary = pd.DataFrame({'features': current_features, 'mi': mi_scores})
    mi_df_summary = mi_df_summary.sort_values('mi', ascending=True).reset_index(drop=True)
    feature_ranking = mi_df_summary['features'].tolist()
    return feature_ranking


def sample_with_min_per_class(
    df: pd.DataFrame,
    target_variable: str,
    num_samples: int,
    min_per_class: int,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Selects a random sample from the input DataFrame such that every unique value
    in the specified target variable is represented at least a minimum number of
    times, while ensuring the total sample size matches the requested number.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to sample rows.
    target_variable : str
        The column name representing the target variable, which defines the classes.
    num_samples : int
        The desired total number of samples in the resulting DataFrame.
    min_per_class : int
        The minimum number of samples required for each unique value in the target variable.
    random_state : int, optional
        A seed to control randomness for reproducibility. Default is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the selected random sample with the specified constraints.

    Raises
    ------
    ValueError
        If the requested `num_samples` is insufficient to allocate the required
        `min_per_class` samples to all classes or if there are not enough remaining
        rows to meet the target sample size after ensuring `min_per_class`.
    """
    # 1) For each class, pull min_per_class (or all if there aren't that many)
    pieces = []
    classes = df[target_variable].unique()
    for c in classes:
        grp = df[df[target_variable] == c]
        n_take = min(len(grp), min_per_class)
        pieces.append(grp.sample(n=n_take, random_state=random_state))

    head = pd.concat(pieces)
    n_head = len(head)

    # 2) Check we still have room in the budget
    if n_head > num_samples:
        raise ValueError(
            f"Cannot draw {min_per_class} samples for each of the "
            f"{len(classes)} classes (need at least {n_head} total, "
            f"but num_samples={num_samples})."
        )

    # 3) Sample the remaining from the leftover pool
    remainder = num_samples - n_head
    pool = df.drop(head.index)
    if remainder > len(pool):
        raise ValueError(
            f"Not enough leftover rows to reach {num_samples} samples "
            f"(only {n_head + len(pool)} available)."
        )
    tail = pool.sample(n=remainder, random_state=random_state)

    # 4) Concatenate and reshuffle
    result = pd.concat([head, tail]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result


def evaluate_feature_set(
        df: pd.DataFrame,
        feature_subset: Tuple[str],
        target_variable: str,
        cv_results: Dict,
        n_neighbors_option: List[int],
        weights_options: List[str],
        cv: StratifiedKFold,
        categorical_features: List[str],
        numerical_features: List[str],
        n_jobs: int,
        logger: logging.Logger
) -> Tuple[float, Tuple, Dict, Dict]:
    """
    Performs hyperparameter optimization for a K-Nearest Neighbors (KNN) classifier
    using Stratified K-Fold Cross-Validation (CV) over a given subset of features.
    It evaluates combinations of hyperparameters (e.g., number of neighbors and weighting function),
    processes categorical and numerical features, and returns the best-performing configuration
    and associated results.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing feature and target data.
    feature_subset : Tuple[str]
        A tuple of feature names to be used for the evaluation.
    target_variable : str
        The name of the target variable to be predicted by the KNN classifier.
    cv_results : Dict
        A dictionary to store the cross-validation results for each configuration.
    n_neighbors_option : List[int]
        A list of possible values for the number of neighbors (k) in the KNN classifier.
    weights_options : List[str]
        A list of weighting functions to use in the KNN classifier (e.g., 'uniform', 'distance').
    cv : StratifiedKFold
        Stratified K-Fold cross-validation object to handle train/test splitting.
    categorical_features : List[str]
        A list of feature names regarded as categorical.
    numerical_features : List[str]
        A list of feature names regarded as numerical.
    n_jobs : int
        The number of parallel jobs to run for the KNN classifier.
    logger: logging.Logger

    Returns
    -------
    Tuple[float, Tuple, Dict, Dict]
        - The best mean cross-validation accuracy obtained (float).
        - A tuple representing the best configuration (selected features, number of neighbors, weights).
        - A dictionary containing detailed results for the best configuration.
        - The full cross-validation results dictionary for all evaluated configurations.
    """
    best_mean_acc = -np.inf
    best_config = None
    best_config_details = None

    # Prepare the data using the given feature subset
    X_full = df[list(feature_subset)]
    y_full = df[target_variable]

    for n_neighbors in n_neighbors_option:
        for weights in weights_options:
            config_key = (tuple(sorted(feature_subset)), n_neighbors, weights)
            fold_accuracies = []
            fold_confusion_matrices = []

            # Perform CV
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_full, y_full), 1):
                try:
                    # Extracting train and test dataframes for X and y
                    X_train_fold = X_full.iloc[train_idx].reset_index(drop=True)
                    y_train_fold = y_full.iloc[train_idx].reset_index(drop=True)
                    X_val_fold = X_full.iloc[val_idx].reset_index(drop=True)
                    y_val_fold = y_full.iloc[val_idx].reset_index(drop=True)

                    # Determining which features are numerical vs categorical in this subset
                    selected_numerical = [col for col in feature_subset if col in numerical_features]
                    selected_categorical = [col for col in feature_subset if col in categorical_features]

                    # Processing numerical features if any
                    if selected_numerical:
                        X_train_numerical, X_val_numerical, X_numerical_scaler = preprocess_numerical_features(
                            train_df=X_train_fold,
                            test_df=X_val_fold,
                            numerical_features=selected_numerical
                        )
                    else:
                        X_train_numerical = csr_matrix((X_train_fold.shape[0], 0))
                        X_val_numerical = csr_matrix((X_val_fold.shape[0], 0))

                    # Process categorical features if any
                    if selected_categorical:
                        X_train_categorical, X_val_categorical, categorical_encoder = preprocess_categorical_features(
                            train_df=X_train_fold,
                            test_df=X_val_fold,
                            categorical_features=selected_categorical
                        )
                    else:
                        X_train_categorical = csr_matrix((X_train_fold.shape[0], 0))
                        X_val_categorical = csr_matrix((X_val_fold.shape[0], 0))

                    # Combine numerical and categorical features
                    X_train_combined = hstack([X_train_numerical, X_train_categorical])
                    X_val_combined = hstack([X_val_numerical, X_val_categorical])

                    # Create and fit the KNN classifier with current hyperparameters
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                                               weights=weights,
                                               n_jobs=n_jobs)
                    knn.fit(X_train_combined, y_train_fold)

                    # Predict and evaluate
                    y_val_pred = knn.predict(X_val_combined)
                    acc = accuracy_score(y_val_fold, y_val_pred)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning,
                                                message=".*single label.*confusion matrix.*")
                        cm = confusion_matrix(y_true=y_val_fold, y_pred=y_val_pred, labels=np.unique(y_full))
                    fold_accuracies.append(acc)
                    fold_confusion_matrices.append(cm)

                except Exception as e:
                    logger.error(f"Error in fold {fold} for config {config_key}: {e}")
                    continue

            if fold_accuracies:
                mean_acc = np.mean(fold_accuracies)
            else:
                mean_acc = 0.0

            # Store the detailed configuration result
            cv_results[config_key] = {
                'fold_accuracies': fold_accuracies,
                'fold_conf_matrices': [cm.tolist() for cm in fold_confusion_matrices],
                'mean_accuracy': mean_acc,
                'num_folds': len(fold_accuracies),
            }

            logger.debug(f"Evaluated config {config_key}: Mean Accuracy = {mean_acc:.4f}")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_config = config_key
                best_config_details = cv_results[config_key]

    return best_mean_acc, best_config, best_config_details, cv_results


def backward_elimination_with_cross_validation(
        df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        target_variable: str,
        must_have_features: List[str],
        n_neighbors_option: List[int],
        weights_options: List[str],
        accuracy_tolerance: float = 0.05,
        n_jobs: int = -1,
        early_stopping_iterations: int = 10,
        sampling_for_efficiency: bool = True,
        num_samples_for_efficiency: int = 10_000,
        random_seed: int = 42,
        logger: logging.Logger = None
) -> Dict:
    """
    Performs backward elimination with cross-validation to identify the optimal subset of features
    for classification using a k-Nearest Neighbors (k-NN) model. The function iteratively removes
    features based on their impact on cross-validated classification accuracy, while leveraging
    hyperparameter tuning for k-NN.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing features and the target variable.
    categorical_features : List[str]
        List of categorical feature column names in the dataset.
    numerical_features : List[str]
        List of numerical feature column names in the dataset.
    target_variable : str
        Name of the target variable column in the dataset.
    must_have_features : List[str]
        List of feature names that must be included in the selected features.
    n_neighbors_option : List[int]
        A list of neighbor values (`n_neighbors`) to consider for k-NN.
    weights_options : List[str]
        A list of weighting schemes (`uniform` or `distance`) to evaluate for k-NN.
    accuracy_tolerance : float, optional, default=0.05
        The threshold under which feature subsets with performance differences from the
        global best can still be considered viable.
    n_jobs : int, optional, default=-1
        Number of parallel jobs to use for computation. `-1` uses all available processors.
    early_stopping_iterations : int, optional, default=10
        Number of iterations without improvement before stopping the backward elimination process.
    sampling_for_efficiency : bool, optional, default=False
        If True, samples a subset of rows for evaluation to improve efficiency in large datasets.
    num_samples_for_efficiency : int, optional, default=10_000
        Number of samples to draw in case sampling for efficiency is True.
    random_seed : int, optional, default=42
        Random seed for reproducibility of the cross-validation process.
    logger: logging.Logger, default=None

    Returns
    -------
    Dict
        A dictionary with the following keys:
        - 'cv_results' : Dict
            Cross-validation results containing performance metrics for all evaluated configurations.
        - 'global_best_config' : Tuple
            The hyperparameter and feature subset configuration achieving the best accuracy.
        - 'global_best_accuracy' : float
            The highest cross-validated accuracy achieved.
        - 'global_best_details' : Dict
            Additional details about the performance of the best configuration.
    """
    CV_FOLDS = 5
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_seed)

    feature_ranking = compute_features_target_mi(df=df,
                                                 categorical_features=categorical_features,
                                                 numerical_features=numerical_features,
                                                 target_variable=target_variable)
    cv_results = {}

    # sampling rows of df for backward elimination efficiency
    if sampling_for_efficiency and len(df) > num_samples_for_efficiency:
        df = sample_with_min_per_class(df=df,
                                       target_variable=target_variable,
                                       num_samples=num_samples_for_efficiency,
                                       min_per_class=5,
                                       random_state=random_seed)

    # Running the evaluation on the entire feature list
    current_features = tuple(sorted(numerical_features + categorical_features))
    eliminated_feature = None
    # Evaluate the model using all features and save the global best performance and configuration
    global_best_accuracy, global_best_config, global_best_details, cv_results = evaluate_feature_set(
        df=df,
        feature_subset=current_features,
        target_variable=target_variable,
        cv_results=cv_results,
        n_neighbors_option=n_neighbors_option,
        weights_options=weights_options,
        cv=cv,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        n_jobs=n_jobs,
        logger=logger
    )
    logger.info(f"Starting backward elimination with full-feature set: "
                f"{current_features} | N_Neighbors: {global_best_config[1]} | Weights: {global_best_config[2]} | "
                f"Best CV Accuracy: {global_best_accuracy:.4f}")
    logger.info(f"Must-have features that will be preserved: {must_have_features}")

    # Begin backward elimination with optimization
    can_continue = True
    iteration = 1
    no_improvement_count = 0
    # stop when only must_have_features remain or when we reach the minimum viable size
    min_features_threshold = max(1, len(must_have_features))

    while can_continue and len(current_features) > min_features_threshold:
        logger.info(f"======================== Iteration {iteration} ======================== ")
        logger.info(f"Global Best Accuracy: {global_best_accuracy}")
        logger.info(f"Eliminated feature in previous iteration: {eliminated_feature}")
        logger.info(f"Current feature set with size {len(current_features)}")
        logger.info(f"{current_features}")

        # Reset continuation flag for current iteration
        can_continue = False
        iteration_improved = False

        # Initialize candidate best with current values
        candidate_best_accuracy = global_best_accuracy
        candidate_best_config = global_best_config
        candidate_best_features = current_features

        # Prioritize features for elimination based on feature importance if available
        # Get the features still in our current set, excluding must_have_features
        remaining_features = set(current_features) - set(must_have_features)

        # Check if we have any eliminable features left
        if not remaining_features:
            logger.info("Only must-have features remain. Stopping elimination process.")
            break

        # Order by importance (least important first)
        ordered_features = [f for f in feature_ranking if f in remaining_features]
        # Add any features not in the ranking (just in case)
        ordered_features.extend([f for f in remaining_features if f not in ordered_features])
        features_to_test = ordered_features

        # Try dropping features (consider parallel processing here if possible)
        feature_results = []
        for feature in features_to_test:
            t0 = time.time()
            candidate_features = tuple(sorted(set(current_features) - {feature}))
            candidate_acc, candidate_config, candidate_details, cv_results = evaluate_feature_set(
                df=df,
                feature_subset=candidate_features,
                target_variable=target_variable,
                cv_results=cv_results,
                n_neighbors_option=n_neighbors_option,
                weights_options=weights_options,
                cv=cv,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                n_jobs=n_jobs,
                logger=logger
            )

            feature_results.append(
                (feature, candidate_features, candidate_acc, candidate_config, candidate_details))
            t1 = time.time()
            logger.info(f"-------- Dropping '{feature}' achieved CV Accuracy: "
                        f"{candidate_acc} | N_Neighbors: {candidate_config[1]} | Weight: {candidate_config[2]} | "
                        f"time: {t1 - t0:.1f}")

        # Process results
        for feature, candidate_features, candidate_acc, candidate_config, candidate_details in feature_results:
            # Check if better OR within tolerance
            if candidate_acc >= candidate_best_accuracy:
                candidate_best_accuracy = candidate_acc
                candidate_best_config = candidate_config
                candidate_best_features = candidate_features
                can_continue = True
                iteration_improved = True
                break
            elif candidate_acc >= (global_best_accuracy - accuracy_tolerance):
                if candidate_acc > candidate_best_accuracy:
                    candidate_best_accuracy = candidate_acc
                    candidate_best_config = candidate_config
                    candidate_best_features = candidate_features
                can_continue = True

        # If a viable candidate was found
        if can_continue:
            diff = list(set(current_features).difference(set(candidate_best_features)))
            if len(diff) == 1:
                eliminated_feature = diff[0]
                current_features = candidate_best_features
                eliminated_feature_acc = candidate_best_accuracy
            else:
                sorted_feature_results = sorted(feature_results, key=lambda x: x[2])
                eliminated_feature = sorted_feature_results[0][0]
                current_features = sorted_feature_results[0][1]
                eliminated_feature_acc = sorted_feature_results[0][2]
            logger.info(f"Dropping '{eliminated_feature}'.")

            # Only update global best if the candidate is actually better
            if candidate_best_accuracy > global_best_accuracy:
                global_best_accuracy = candidate_best_accuracy
                global_best_config = candidate_best_config
                global_best_details = cv_results[global_best_config]
                logger.info(f"Iteration {iteration} found improvement. New best accuracy: {global_best_accuracy}")
                no_improvement_count = 0  # Reset early stopping counter
            else:
                # If within tolerance but not better
                logger.info(f"Iteration {iteration} found viable feature set within tolerance {accuracy_tolerance}. "
                            f"Eliminated Feature Accuracy: {eliminated_feature_acc} "
                            f"(global best: {global_best_accuracy})")

                # Update early stopping counter
                if not iteration_improved:
                    no_improvement_count += 1
                    logger.info(f"No improvement in accuracy for {no_improvement_count} iterations")

                    # Check early stopping condition
                    if no_improvement_count >= early_stopping_iterations:
                        logger.info(
                            f"Early stopping triggered after {early_stopping_iterations} iterations "
                            f"without improvement")
                        can_continue = False
        else:
            logger.info(f"Iteration {iteration} found no viable improvements. Stopping.")

        iteration += 1

    backward_elimination_info = {
        'cv_results': cv_results,
        'global_best_config': global_best_config,
        'global_best_accuracy': global_best_accuracy,
        'global_best_details': global_best_details
    }
    return backward_elimination_info


def postprocess_cv_results(
        cv_results: Dict,
) -> Tuple[List[Dict], Dict]:
    """
    Post-process cross-validation results and select the best configuration.

    This function processes the raw cross-validation results into a sorted list of configurations
    with their corresponding metadata. It then identifies the best configuration by considering the
    highest mean accuracy within a specified tolerance and the configuration with the smallest
    number of features in the top group.

    Parameters
    ----------
    cv_results : dict
        A dictionary of cross-validation results where each key is a configuration represented
        as a tuple (features, n_neighbors, weights) and each value contains corresponding performance
        metrics (e.g., mean_accuracy).

    Returns
    -------
    processed_cv_results : list of dict
        A sorted list of dictionaries where each dictionary represents a configuration and its
        associated metrics. The list is sorted in descending order by mean accuracy.

    cv_best_minimal_config : dict
        The best configuration dictionary selected based on the highest mean accuracy within a
        given tolerance and the smallest number of features among top configurations.
    """
    processed_cv_results = []
    for config, config_data in cv_results.items():
        processed_cv_results.append(
            {
                "features": list(config[0]),
                "n_neighbors": config[1],
                "weights": config[2],
                **config_data
            }
        )
    processed_cv_results = sorted(processed_cv_results, key=lambda x: x['mean_accuracy'], reverse=True)

    # selecting the best config
    tolerance = 0.01
    top_accuracy = processed_cv_results[0]['mean_accuracy']
    top_group = [result for result in processed_cv_results if (top_accuracy - result['mean_accuracy']) <= tolerance]
    cv_best_minimal_config = min(top_group, key=lambda result: len(result['features']))

    return processed_cv_results, cv_best_minimal_config
