import pandas as pd
from config.config import TRAIN_FILE, TEST_FILE, LABEL_MAPPING, RANDOM_SEED, VAL_SPLIT
from sklearn.model_selection import train_test_split

def load_training_data():
    df = pd.read_csv(TRAIN_FILE, index_col=0)
    df["label_encoded"] = df["label"].map(LABEL_MAPPING)
    return df

def load_test_data():
    df = pd.read_csv(TEST_FILE, index_col=0)
    return df

def create_train_val_split(df):
    sentences = df["sentence"].to_frame()
    labels = df["label_encoded"].to_frame()
    return train_test_split(
        sentences,
        labels,
        test_size=VAL_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
