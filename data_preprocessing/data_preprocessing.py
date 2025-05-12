import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#For Stemming text
from nltk.stem import PorterStemmer

import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Convert a collection of text documents to a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer

#For evaluation of model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# nltk.download('vader_lexicon')
# nltk.download('punkt_tab')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

sid = SentimentIntensityAnalyzer()
stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()

import re
import unicodedata
import re

def minimal_llm_preprocess(df):
    df = df.copy()

    # Prevent modifications to the original DataFrame
    df = df.copy()

    # 1. Lowercase the text
    df['text'] = df['text'].apply(lambda x: str(x).lower())

    # Remove emails
    df['text'] = df['text'].apply(
        lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', "", x))

    #  Remove URLs
    df['text'] = df['text'].apply(
        lambda x: re.sub(
            r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
            "", x))

    return df

def nlp_analysis(df):
    """
    Perform basic NLP feature extraction on a DataFrame containing text data.

    This function augments the input DataFrame by computing several text-based
    features for each sentence in the 'sentence' column, including word counts,
    character counts, average word length, stopword counts, numerical counts,
    uppercase word counts, email detection, and URL detection.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame with at least a 'sentence' column containing text data.

    Returns:
    -------
    df : pandas.DataFrame
        The original DataFrame augmented with the following new columns:
        - 'word_counts': Number of words in each sentence
        - 'char_counts': Number of characters in each sentence
        - 'avg_word_len': Average word length
        - 'stop_words_len': Number of stopwords
        - 'sentence_no_stop': Sentence without stopwords
        - 'numerics_count': Number of numeric tokens
        - 'upper_counts': Number of fully uppercase words
        - 'emails': List of email addresses found in the sentence
        - 'urls': Number of URLs found in the sentence
    """
    # Prevent modifications to the original DataFrame
    df = df.copy()

    # Word count per sentence
    df['word_counts'] = df['sentence'].apply(lambda x: len(str(x).split()))

    # Character count per sentence
    df['char_counts'] = df['sentence'].apply(
        lambda x: char_counts(str(x)))  # assumes custom `char_counts` function is defined

    # Average word length
    df['avg_word_len'] = df['char_counts'] / df['word_counts']

    # Count of stopwords in each sentence
    df['stop_words_len'] = df['sentence'].apply(
        lambda x: len([t for t in x.split() if t in stopwords]))  # assumes `stopwords` set is defined

    # Sentence after removing stopwords
    df['sentence_no_stop'] = df['sentence'].apply(
        lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

    # Count of numeric words in each sentence
    df['numerics_count'] = df['sentence'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))

    # Count of fully uppercase words
    df['upper_counts'] = df['sentence'].apply(lambda x: len([t for t in x.split() if t.isupper()]))

    # Extract email addresses using regex
    df['emails'] = df['sentence'].apply(
        lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x, re.I))

    # Count number of URLs using regex
    df['urls'] = df['sentence'].apply(lambda x: len(
        re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

    return df


def nlp_preprocess(df):
    """
    Perform standard NLP preprocessing on the 'sentence' column of a DataFrame.

    This function applies several text cleaning and normalization steps:
    1. Converts all text to lowercase.
    2. Expands common English contractions (e.g., "don't" → "do not").
    3. Removes email addresses.
    4. Removes URLs.
    5. Removes all numeric characters.
    6. Removes special characters and punctuation.
    7. Collapses multiple spaces into a single space.
    8. Removes accented characters (e.g., "é" → "e").
    9. Correct spelling errors
    10. Tokenizes the sentence into words.
    11. Removes common stopwords (e.g., "the", "is").
    12. Lemmatizes each word to its base/dictionary form.
    13. Reforms the sentence using cleaned tokens that are either valid words or non-alphabetic.

    Parameters:
        df (pd.DataFrame): A DataFrame with a 'sentence' column containing text data.

    Returns:
        pd.DataFrame: The input DataFrame with the 'sentence' column cleaned and preprocessed.
    """
    # Prevent modifications to the original DataFrame
    df = df.copy()

    # 1. Lowercase the text
    df['sentence'] = df['sentence'].apply(lambda x: str(x).lower())

    # 2. Expand contractions (e.g., "don't" -> "do not")
    df['sentence'] = df['sentence'].apply(lambda x: cont_to_exp(x))

    # 3. Remove emails
    df['sentence'] = df['sentence'].apply(
        lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', "", x))

    # 4. Remove URLs
    df['sentence'] = df['sentence'].apply(
        lambda x: re.sub(
            r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
            "", x))

    # 5. Remove digits
    df['sentence'] = df['sentence'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))

    # 6. Remove special characters and punctuation (replace with space)
    df['sentence'] = df['sentence'].apply(lambda x: re.sub(r'[^\w]+', " ", x))

    # 7. Collapse multiple spaces into one
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join(x.split()))

    # 8. Remove accented characters (e.g., "é" → "e")
    df['sentence'] = df['sentence'].apply(lambda x: remove_accented_chars(x))

    # # 9. Correct spelling errors - SLOW !!! -> 40 sec for 1000 samples -> would take around 76min in total for train, valid and test sets
    # df['sentence'] = df['sentence'].apply(lambda x: TextBlob(x).correct().raw)

    # 10. Tokenize text into a list of words
    df['sentence'] = df['sentence'].apply(lambda x: nltk.tokenize.word_tokenize(x))

    # 11. Remove stopwords
    df['sentence'] = df['sentence'].apply(lambda x: [i for i in x if i not in stopwords])

    # 12. Lemmatize each word
    df['sentence'] = df['sentence'].apply(lambda x: (lemmatizer.lemmatize(w) for w in x))

    # 13. Reform the sentence with valid lemmatized words or non-alphabetic tokens
    df['sentence'] = df['sentence'].apply(
        lambda x: ' '.join(w for w in x if w.lower() in words or not w.isalpha()))

    return df

def nlp_preprocess_train_valid_test(train, valid, test):
    """
    Apply standard NLP preprocessing to the train, validation, and test DataFrames.

    This function takes three datasets (train, valid, test), applies the `nlp_preprocess`
    function to each, and returns the processed versions.

    Parameters:
        train (pd.DataFrame): The training dataset containing a 'sentence' column.
        valid (pd.DataFrame): The validation dataset containing a 'sentence' column.
        test (pd.DataFrame): The test dataset containing a 'sentence' column.

    Returns:
        tuple: A tuple of three DataFrames: (train_cleaned, valid_cleaned, test_cleaned),
               each having the 'sentence' column preprocessed using `nlp_preprocess`.

    Notes:
        - Assumes the presence of a column named 'sentence' in each input DataFrame.
        - The `nlp_preprocess` function must be defined in the current context.
    """
    return nlp_preprocess(train), nlp_preprocess(valid), nlp_preprocess(test)

# TODO -> can improve this dataset (or maybe try feed into LLM)
contractionhh = {
    "a'ight": "alright",
    "ain't": "am not",
    "amn't": "am not",
    "aren't": "are not",
    "‘bout": "about",
    "can't": "cannot",
    "cap’n": "captain",
    "'cause": "because",
    "’cept": "except",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "dammit": "damn it",
    "daren't": "dare not ",
    "daresn't": "dare not",
    "dasn't": "dare not",
    "didn't": "did not",
    "don't": "do not",
    "he 'll": "he will ",
    "hadn't": "had not",
    "hadn't": "had not have",
    "has't": "has not",
    "have't": "have not",
    "he'd": "he would",
    "he'll": "he will ",
    "he'll've": "he will	have",
    "he 's": "he is",
    "how 'd ": "how d id",
    "how'd'y ": "how do you ",
    "how'll": "how will",
    "how's": "f how does",
    "i'd": "i would ",
    "i'd've": "i would	have",
    "i'll": "i will",
    "i'll've": "i will	have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd ": "it would ",
    "it'd've ": "it wou ld have",
    "it'll": "it will ",
    "it'll ' ve": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam ",
    "mayn't ": "may not ",
    "might've": "might	have",
    "mightn't ": "might	not",
    "must've": "must have",
    "mustn't ": "must not ",
    "mustn't've": "mu st not	have",
    "needn't ": "need not ",
    "needn't ' ve": "need	not	have",
    "o'clock": "of	the clock ",
    "oughtn't ": "ought	not ",
    "oughtn't ' ve": "ought not have",
    "shan't": "shall not ",
    "sha'n't ": "shall not ",
    "shan't've": "sha ll not",
    "she'd": "she would ",
    "she'd've ": "she would",
    "she'll": "she will ",
    "she'll've": "she will",
    "she's": "she is",
    "should've": "should	have",
    "shouldn't ": "should	not",
    "shouldn't've": "should	not	have",
    "so've": "so have",
    "so's": "so  is",
    "that'd": "that	wou ld",
    "that'd ' ve": "that wou ld	have",
    "that's": "that	is",
    "there'd": "there would",
    "there'd've": "the re wou ld	have",
    "there's": "there	is",
    "they'd": "they would",
    "they'd've": "they wou ld	have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "t hey are",
    "they've": "they have",
    "to've": "to have",
    "wasn't ": " was not ",
}

def cont_to_exp(x):
    if type(x) is str:
        for key in contractionhh:
            value = contractionhh[key]
            x = x.replace(key, value)
        return x
    else:
        return x

def remove_accented_chars(x):
  x = unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
  return x

def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)
