"""
Contains functions to load and preprocess data for training, validation, and testing.
"""

import pandas as pd
from config.config import (
    TRAIN_FILE,
    TEST_FILE,
    LABEL_MAPPING_STRING_TO_NUMBER,
    RANDOM_SEED,
    VAL_TEST_SPLIT,
)
from sklearn.model_selection import train_test_split


def load_and_split_data(
    file_path=TRAIN_FILE, test_size=VAL_TEST_SPLIT, seed=RANDOM_SEED
):
    """
    Load dataset from a CSV file, encode labels, and split into train, validation, and test sets.

    Args:
        file_path (str): Path to the CSV file containing the training data.
        test_size (float): Fraction of the data to use for validation and test sets.
        seed (int): Random seed for reproducibility of the splits.

    Returns:
        tuple: A tuple containing three pandas DataFrames (train_df, val_df, test_df),
            each with two columns:
                - 'text': the sentence text
                - 'labels': the encoded label as an integer
    """
    # Load data
    training_data = pd.read_csv(file_path, index_col=0)

    # Encode labels
    training_data["label_encoded"] = training_data["label"].map(
        LABEL_MAPPING_STRING_TO_NUMBER
    )

    # Extract sentences and labels
    sentences = training_data["sentence"]
    labels = training_data["label_encoded"]

    # First split into training and temp sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    # Then split the temp set equally into validation and test sets
    val_sentences, test_sentences, val_labels, test_labels = train_test_split(
        val_sentences, val_labels, test_size=0.5, stratify=val_labels, random_state=seed
    )

    # Create DataFrames
    train_df = pd.DataFrame({"text": train_sentences, "labels": train_labels})
    val_df = pd.DataFrame({"text": val_sentences, "labels": val_labels})
    test_df = pd.DataFrame({"text": test_sentences, "labels": test_labels})

    return train_df, val_df, test_df


def load_submission_data(file_path=TEST_FILE):
    """
    Load submission data from a CSV file and prepare it for model inference.

    This function reads the submission dataset, extracts the sentence text,
    and assigns placeholder labels (all zeros).

    Args:
        file_path (str): Path to the CSV file containing the test/submission data.

    Returns:
        pandas.DataFrame: A DataFrame with two columns:
            - 'text': the sentence text
            - 'labels': placeholder labels (zeros)
    """
    # Load data
    submission_data = pd.read_csv(file_path, index_col=0)

    # Extract sentences
    sentences = submission_data["sentence"]

    # Create DataFrame with placeholder labels
    submission_df = pd.DataFrame({"text": sentences, "labels": [0] * len(sentences)})

    return submission_df
