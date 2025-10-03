import json
import pandas as pd
from typing import Dict, List


def generate_gst_validation_system_prompt(
        feature_list: List[str],
        unique_tax_types: List[str]
) -> str:
    """
    Generate a system prompt for GST validation classification using LLM.

    Creates a comprehensive system prompt that instructs an LLM to classify journal
    transactions into appropriate tax types based on similar historical records
    and contextual features.

    Parameters
    ----------
    feature_list : List[str]
        List of feature names to be used for finding similar records. Features like
        'description', 'contact_name', and 'journal_date' are handled specially.
    unique_tax_types : List[str]
        List of all available tax types that the LLM must choose from for
        classification. These are the only valid tax types for the output.

    Returns
    -------
    str
        A formatted system prompt string that provides detailed instructions for
        GST validation classification, including available tax types,
        classification criteria, and required JSON output format.
    """
    feature_sections = []
    for feature in feature_list:
        if feature.lower() in ['description', 'contact_name', 'journal_date']:
            continue
        feature_sections.append(
            f"- **Similar records by {feature}**: Top most similar rows with comparable {feature} values")
    feature_sections_str = "\n".join(feature_sections) if feature_sections else ""

    tax_types_formatted = "\n".join([f"  - {name}" for name in unique_tax_types])

    system_prompt = f"""
    You are an expert agent specializing in GST validation and tax type classification for journal entries. 
    Your task is to classify the correct tax_type for a given journal transaction row based on 
    contextual information and similar historical records.

    You will receive the following information:
    - **Available tax types**: A complete list of all unique tax_type values from which you must select
    - **Query row**: The journal transaction row that needs to be classified
    - **Similar records by description**: Top most similar rows with comparable transaction descriptions
    - **Similar records by contact_name**: Top most similar rows with comparable contact names
    - **Similar records by journal_date**: Top most similar rows with comparable journal dates
    {feature_sections_str}

    ## Unique Tax Types

    The following are all the available tax types from which you must select the correct classification:

    {tax_types_formatted}

    You MUST select one of these exact tax types for your classification. 
    Do not create new tax types or modify existing ones.

    ## Your Classification Process

    1. **Analyze the query row**: Review all available information about the transaction requiring classification
    2. **Examine similar records**: Study the patterns in similar historical transactions and their assigned tax types
    3. **Identify key indicators**: Look for consistent patterns across transaction descriptions, contact names,
     dates, and other features
    4. **Apply domain knowledge**: Use GST regulations and tax compliance logic to determine the most 
    appropriate tax type classification
    5. **Make classification decision**: Select the tax_type that best matches the transaction characteristics

    ## Classification Criteria

    - **Transaction description patterns**: Analyze keywords, vendor names, expense types, 
    and transaction purposes to determine tax implications
    - **Contact name consistency**: Consider how similar contacts have been classified historically for tax purposes
    - **Temporal patterns**: Account for seasonal variations or date-related GST classification trends
    - **Feature correlation**: Evaluate how other transaction attributes relate to tax type classifications
    - **Historical precedent**: Prioritize classifications that align with similar past transactions
    - **Tax regulations**: Apply GST rules and tax compliance requirements to ensure correct classification

    ## Output Requirements

    You must provide your response in the following JSON format:
    ```json
    {{
        "target_variable": "selected_tax_type",
        "reasoning": "detailed_explanation_of_classification_decision",
        "confidence_score": 0.0_to_1.0
    }}
    ```
    Where:
    - "target_variable": The exact tax type selected from the provided list of available 
    tax types `tax_type`
    - "reasoning": A clear 2-3 sentence explanation of why this tax type was chosen, 
    referencing specific evidence from the query row and similar records `reasoning`
    - "confidence_score": A float between 0.0 and 1.0 indicating your certainty in 
    the classification (1.0 = completely certain, 0.0 = completely uncertain) `confidence_score`

    ## Important Guidelines

    - You MUST select a tax_type from the provided list of available tax types
    - Base your decision on evidence from the query row and similar historical records
    - Provide specific reasoning that references the key factors that influenced your decision
    - Be honest about uncertainty - lower confidence scores for ambiguous cases are acceptable
    - Consider the consistency of historical classifications when making your decision
    - Apply GST regulations and tax compliance knowledge to ensure accurate classification
    """
    return system_prompt


def generate_gst_validation_user_prompt(
        query_df: pd.DataFrame,
        similar_records_by_feature: Dict[str, pd.DataFrame]
) -> str:
    """
    Generate a user prompt for LLM-based GST validation classification.

    Creates a formatted prompt containing a query transaction record and similar historical
    records organized by feature type. The prompt instructs the LLM to classify the query
    record with the correct tax type based on patterns in similar historical transactions.

    Parameters
    ----------
    query_df : pd.DataFrame
        DataFrame containing a single transaction record to be classified. The first row
        is used as the query record.
    similar_records_by_feature : Dict[str, pd.DataFrame]
        Dictionary mapping feature names to DataFrames containing similar historical records
        for each feature. Each DataFrame contains records that are most similar to the query
        record based on the corresponding feature.

    Returns
    -------
    str
        Formatted user prompt containing the query record in JSON format, similar historical
        records organized by feature type, and detailed instructions for the LLM to perform
        GST validation classification with required output format specification.
    """
    query_row = query_df.iloc[0].to_dict()
    query_json = json.dumps(query_row, indent=2, default=str)

    similar_records_sections = []
    for feature_name, similar_df in similar_records_by_feature.items():
        similar_records = similar_df.to_dict('records')
        similar_records_json = json.dumps(similar_records, indent=2, default=str)
        similar_records_text = f"""
        ### Similar Records by {feature_name}:
        ```json
        {similar_records_json}
        ```"""
        similar_records_sections.append(similar_records_text)
    similar_records_content = "\n".join(similar_records_sections)

    user_prompt = f"""
    I will provide you with a journal transaction row that needs to be classified with the correct tax_type, 
    along with similar historical records that can help guide your decision.

    **Your task:**
    Based on the information provided, classify the correct tax_type for the query row as described in 
    your system instructions. 
    You must select from the available tax types list and provide your reasoning and confidence score in 
    the exact JSON format specified.

    **Query Row to Classify:**

    ```json
    {query_json}
    ```

    **Similar Historical Records:** 

    The following are the most similar historical records for different features that can help guide your 
    classification decision: 

    {similar_records_content}

    **Instructions:**
    1. Analyze the query row and examine the patterns in the similar historical records
    2. Look for consistent tax type assignments across similar transactions
    3. Consider transaction descriptions, contact names, dates, and other relevant features
    4. Apply GST regulations and tax compliance knowledge to determine the appropriate tax type
    5. Select the most appropriate tax_type from the available options
    6. Provide your response in the exact JSON format specified in your system instructions

    **Required Output Format:**
    {{
        "target_variable": "selected_tax_type",
        "reasoning": "detailed_explanation_of_classification_decision",
        "confidence_score": 0.0_to_1.0
    }}

    Analyze the data above and provide your classification decision.
    """
    return user_prompt
