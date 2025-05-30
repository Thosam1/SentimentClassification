from langdetect import detect, DetectorFactory
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


from config.config import RANDOM_SEED
from utils.utils import get_device

DetectorFactory.seed = RANDOM_SEED


# Detect language for each text in train_df, handle empty strings
def safe_detect(text):
    text = str(text).strip()
    if not text:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def load_language_detection_pipeline(model_name: str):
    """
    Loads a Hugging Face language classification pipeline.

    Args:
        model_name (str): Name of the model on Hugging Face Hub.

    Returns:
        Pipeline: A Hugging Face text classification pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = get_device()
    return pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=device
    )


def add_detected_language_column(
    df: pd.DataFrame,
    text_column: str = "text",
    new_column: str = "transformer_language",
    model_name: str = "qanastek/51-languages-classifier",
) -> pd.DataFrame:
    """
    Adds a new column with language predictions to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a column of text data.
        text_column (str): Name of the column containing the text.
        new_column (str): Name of the column to store predicted language labels.
        model_name (str): Hugging Face model name for language detection.

    Returns:
        pd.DataFrame: The DataFrame with an additional column containing language labels.
    """
    lang_pipeline = load_language_detection_pipeline(model_name)
    tqdm.pandas(desc="Detecting languages")

    def classify(text: str) -> str:
        text = str(text).strip()
        if not text:
            return "unknown"
        try:
            result = lang_pipeline(text, truncation=True, max_length=512)
            return result[0]["label"]
        except Exception:
            return "unknown"

    df[new_column] = df[text_column].progress_apply(classify)
    return df
