from sklearn.metrics import classification_report, mean_absolute_error

from config.config import LABEL_MAPPING_STRING_TO_NUMBER


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
