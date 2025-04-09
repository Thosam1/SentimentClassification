from pathlib import Path

DATA_DIR = Path("../data")
TRAIN_FILE = DATA_DIR / "training.csv"
TEST_FILE = DATA_DIR / "test.csv"

LABEL_MAPPING = {'negative': -1, 'neutral': 0, 'positive': 1}

RANDOM_SEED = 42
VAL_SPLIT = 0.1
