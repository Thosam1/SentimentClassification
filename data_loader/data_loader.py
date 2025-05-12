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

def load_and_split_data(file_path='../data/training.csv', test_size=0.2, seed=42):
    # Load data
    training_data = pd.read_csv(file_path, index_col=0)

    # Encode labels
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    training_data['label_encoded'] = training_data['label'].map(label_mapping)

    # Extract sentences and labels
    sentences = training_data['sentence']
    labels = training_data['label_encoded']

    # Split data
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    val_sentences, test_sentences, val_labels, test_labels = train_test_split(
        val_sentences,
        val_labels,
        test_size=0.5,
        stratify=val_labels,
        random_state=seed
    )

    # Create DataFrames
    train_df = pd.DataFrame({"text": train_sentences, "labels": train_labels})
    val_df = pd.DataFrame({"text": val_sentences, "labels": val_labels})
    test_df = pd.DataFrame({"text": test_sentences, "labels": test_labels})

    return train_df, val_df, test_df

def load_submission_data(file_path='../data/test.csv'):
    # Load data
    submission_data = pd.read_csv(file_path, index_col=0)

    # Extract sentences
    sentences = submission_data['sentence']

    # Create DataFrame with placeholder labels
    submission_df = pd.DataFrame({
        "text": sentences,
        "labels": [0] * len(sentences)
    })

    return submission_df


