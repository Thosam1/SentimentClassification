from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple


def preprocess_text_with_count_vectorizer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "labels",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
) -> Tuple[
    csr_matrix, csr_matrix, csr_matrix, pd.Series, pd.Series, pd.Series, CountVectorizer
]:
    """
    Preprocesses text data using CountVectorizer with specified n-gram range and feature limit.

    Fits the vectorizer on the training data and transforms the validation and test data.
    Also extracts the corresponding labels.

    Args:
        train_df (pd.DataFrame): Training dataset containing text and labels.
        val_df (pd.DataFrame): Validation dataset containing text and labels.
        test_df (pd.DataFrame): Test dataset containing text and labels.
        text_column (str): Column name for the text data. Defaults to "text".
        label_column (str): Column name for the labels. Defaults to "labels".
        ngram_range (Tuple[int, int]): The lower and upper boundary of the n-grams to be extracted.
        max_features (int): Maximum number of features to keep.

    Returns:
        Tuple containing:
            - X_train (csr_matrix): Transformed training text data.
            - X_val (csr_matrix): Transformed validation text data.
            - X_test (csr_matrix): Transformed test text data.
            - y_train (pd.Series): Labels for the training data.
            - y_val (pd.Series): Labels for the validation data.
            - y_test (pd.Series): Labels for the test data.
            - vectorizer (CountVectorizer): Fitted CountVectorizer instance.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_train = vectorizer.fit_transform(train_df[text_column])
    X_val = vectorizer.transform(val_df[text_column])
    X_test = vectorizer.transform(test_df[text_column])

    y_train = train_df[label_column]
    y_val = val_df[label_column]
    y_test = test_df[label_column]

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer





