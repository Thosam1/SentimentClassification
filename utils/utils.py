def decode_label(label_int):
    mapping = {-1: "negative", 0: "neutral", 1: "positive"}
    return mapping[label_int]

def encode_label(label_int):
    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    return mapping[label_int]