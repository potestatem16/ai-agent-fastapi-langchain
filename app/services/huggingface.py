"""
HuggingFace service module.

This module contains functions to load datasets from HuggingFace and
parse the 'reference_information' field.
"""

import ast
from datasets import load_dataset
import pandas as pd

def parse_reference_info(ref):
    """
    Parse the 'reference_information' field.

    It is assumed that each value is a list with a single string that contains
    the JSON representation. This function uses ast.literal_eval to safely evaluate
    the string as a Python literal.

    Args:
        ref: The raw reference information.

    Returns:
        The parsed reference information as a Python object, or None if parsing fails.
    """
    if isinstance(ref, list) and len(ref) == 1 and isinstance(ref[0], str):
        try:
            parsed = ast.literal_eval(ref[0])
            return parsed
        except Exception as e:
            print(f"Error parsing reference info: {ref}. Error: {e}")
            return None
    else:
        try:
            return ast.literal_eval(ref)
        except Exception as e:
            print(f"Error parsing reference info: {ref}. Error: {e}")
            return None

def load_travelplanner_dataset(split="test"):
    """
    Load the 'osunlp/TravelPlanner' dataset from HuggingFace and parse its reference information.

    Args:
        split (str): The dataset split to load. Default is "test".

    Returns:
        A pandas DataFrame with the dataset and an additional column
        'reference_information_parsed' containing the parsed reference info.
    """
    dataset = load_dataset("osunlp/TravelPlanner", split=split)
    df = dataset.to_pandas()
    # Apply the parsing function to the 'reference_information' column
    df['reference_information_parsed'] = df['reference_information'].apply(parse_reference_info)
    return df
