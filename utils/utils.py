import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, mean_absolute_error

from config.config import LABEL_MAPPING_STRING_TO_NUMBER


def get_device():
    """Return the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def decode_label(label_int):
    mapping = {-1: "negative", 0: "neutral", 1: "positive"}
    return mapping[label_int]


def encode_label(label_int):
    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    return mapping[label_int]


def print_evaluation(y_test, y_pred):
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(LABEL_MAPPING_STRING_TO_NUMBER.values()),
            target_names=list(LABEL_MAPPING_STRING_TO_NUMBER.keys()),
        )
    )

def L_score(y_test, y_pred):
    mae_val = mean_absolute_error(y_test, y_pred)
    L_score_val = 0.5 * (2 - mae_val)
    return L_score_val


def classification_breakdown(y_true, y_pred):
    """
    Classifies predictions into:
    - correctly classified
    - partially misclassified (neutral <-> positive or neutral <-> negative)
    - fully misclassified (positive <-> negative)
    """
    correct = 0
    partial = 0
    full = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
        elif abs(yt - yp) == 1:
            partial += 1  # e.g., 0 vs 1 or 0 vs -1
        else:
            full += 1     # e.g., -1 vs 1

    total = correct + partial + full
    print(f"Total samples: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Partially misclassified (neutral <-> pos/neg): {partial}")
    print(f"Misclassified (positive <-> negative): {full}")

    return {"correctly_classified": correct, "partially_misclassified": partial, "fully_misclassified": full}

