from pathlib import Path

DATA_DIR = Path("../data")
TRAIN_FILE = DATA_DIR / "training.csv"
TEST_FILE = DATA_DIR / "test.csv"

LABEL_MAPPING_STRING_TO_NUMBER = {'negative': 0, 'neutral': 1, 'positive': 2}
LABEL_MAPPING_NUMBER_TO_STRING = {0: 'negative', 1: 'neutral', 2: 'positive'}

RANDOM_SEED = 42
VAL_TEST_SPLIT = 0.2
