import sys
from typing import Tuple

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("..")

import pandas as pd
import re
from typing import Callable, List
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("words")
nltk.download("wordnet")

sid = SentimentIntensityAnalyzer()
stopwords = set(nltk.corpus.stopwords.words("english"))
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()


def lowercase(text: str) -> str:
    return text.lower()


def expand_contractions(text: str) -> str:
    return cont_to_exp(text)


def remove_emails(text: str) -> str:
    return re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "", text)


def remove_urls(text: str) -> str:
    return re.sub(r"(http|https|ftp|ssh)://[^\s]+", "", text)


def remove_digits(text: str) -> str:
    return re.sub(r"\d+", "", text)


def remove_special_chars(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def collapse_spaces(text: str) -> str:
    return " ".join(text.split())


def remove_accented_chars(text: str) -> str:
    return unidecode(text)


def correct_spelling(text: str) -> str:
    from textblob import TextBlob

    return str(TextBlob(text).correct())


def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def remove_stopwords(tokens: str) -> str:
    token_list = tokenize(tokens)
    token_list_wo_stopwords = [w for w in token_list if w not in stopwords]
    return " ".join(w for w in token_list_wo_stopwords)


def lemmatize(tokens: str) -> str:
    token_list = tokenize(tokens)
    token_list_lemmatized = [lemmatizer.lemmatize(w) for w in token_list]
    return " ".join(w for w in token_list_lemmatized)


def filter_valid_words(tokens: str) -> str:
    token_list = tokenize(tokens)
    return " ".join(w for w in token_list if w in words or not w.isalpha())


contractions = pd.read_csv("../data/contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))


def cont_to_exp(text):
    for key, value in cont_dic.items():
        text = text.replace(key, value)
    return text


def preprocess_ml_pipeline(
    df: pd.DataFrame, steps: List[str], text_column: str = "text"
) -> pd.DataFrame:
    """
    Apply specified NLP preprocessing steps in order.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        steps (List[str]): List of preprocessing step names to apply in order.
        text_column (str): Name of the column to preprocess.

    Returns:
        pd.DataFrame: New DataFrame with preprocessed text.
    """
    df = df.copy()

    step_functions: dict[str, Callable] = {
        "lowercase": lowercase,
        "expand_contractions": expand_contractions,
        "remove_emails": remove_emails,
        "remove_urls": remove_urls,
        "remove_digits": remove_digits,
        "remove_special_chars": remove_special_chars,
        "collapse_spaces": collapse_spaces,
        "remove_accented_chars": remove_accented_chars,
        # "correct_spelling": correct_spelling,  # slow!
        "remove_stopwords": remove_stopwords,
        "lemmatize": lemmatize,
        "filter_valid_words": filter_valid_words,
    }

    for step in steps:
        if step not in step_functions:
            raise ValueError(f"Unknown preprocessing step: {step}")
        func = step_functions[step]
        df[text_column] = df[text_column].apply(func)

    return df


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

def nlp_analysis(df):
    """
    Perform basic NLP feature extraction on a DataFrame containing text data.

    This function augments the input DataFrame by computing several text-based
    features for each text in the 'text' column, including word counts,
    character counts, average word length, stopword counts, numerical counts,
    uppercase word counts, email detection, and URL detection.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame with at least a 'text' column containing text data.

    Returns:
    -------
    df : pandas.DataFrame
        The original DataFrame augmented with the following new columns:
        - 'word_counts': Number of words in each text
        - 'char_counts': Number of characters in each text
        - 'avg_word_len': Average word length
        - 'stop_words_len': Number of stopwords
        - 'text_no_stop': Text without stopwords
        - 'numerics_count': Number of numeric tokens
        - 'upper_counts': Number of fully uppercase words
        - 'emails': List of email addresses found in the text
        - 'urls': Number of URLs found in the text
    """
    # Prevent modifications to the original DataFrame
    df = df.copy()

    # Word count per text
    df['word_counts'] = df['text'].apply(lambda x: len(str(x).split()))

    # Character count per text
    df['char_counts'] = df['text'].apply(
        lambda x: char_counts(str(x)))  # assumes custom `char_counts` function is defined

    # Average word length
    df['avg_word_len'] = df['char_counts'] / df['word_counts']

    # Count of stopwords in each text
    df['stop_words_len'] = df['text'].apply(
        lambda x: len([t for t in x.split() if t in stopwords]))  # assumes `stopwords` set is defined

    # text after removing stopwords
    df['text_no_stop'] = df['text'].apply(
        lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

    # Count of numeric words in each text
    df['numerics_count'] = df['text'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))

    # Count of fully uppercase words
    df['upper_counts'] = df['text'].apply(lambda x: len([t for t in x.split() if t.isupper()]))

    # Extract email addresses using regex
    df['emails'] = df['text'].apply(
        lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x, re.I))

    # Count number of URLs using regex
    df['urls'] = df['text'].apply(lambda x: len(
        re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    return df

def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)
