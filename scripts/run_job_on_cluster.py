import os
import sys
import random
from dataclasses import dataclass
from collections import defaultdict

# --- Third-Party Libraries ---
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import RANDOM_SEED

#
# # # --- Ensure parent directory is in path ---
# sys.path.append("..")

# --- Local Application/Module Imports ---
from data_loader.data_loader import load_and_split_data, load_submission_data
from data_preprocessing.dataset_dataloader import create_data_loader
from visualizations.visualizations import *
from models.models import *

# --- Environment Config ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    batch_size: int
    model: str
    seed: int
    lr: float
    epochs: int
    dropout: float
    attention_dropout: float
    patience: int
    lora_r: int
    lora_alpha: float
    device: str
    num_classes: int
    max_len: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    device = get_device()

    model_name_list = [
        # "microsoft/deberta-v3-base",
        # "microsoft/deberta-v3-large",
        # "FacebookAI/roberta-base",
        # "FacebookAI/xlm-roberta-base",
        # "FacebookAI/roberta-large",
        # "FacebookAI/xlm-roberta-large",
        # "distilbert/distilbert-base-cased",
        # "distilbert/distilbert-base-multilingual-cased",
    ]

    model_names = [
        # "deberta-v3-base",
        # "deberta-v3-large",
        #"roberta-base",
        # "xlm-roberta-base",
        # "roberta-large",
        # "xlm-roberta-large",
        #"distilbert-base-cased",
        # "distilbert-base-multilingual-cased"
    ]

    for i, model_name in enumerate(model_name_list):

        config = Config(
            batch_size=16,
            model=model_name,
            seed=RANDOM_SEED,
            lr=3e-5,
            epochs=4,
            dropout=0.1,
            attention_dropout=0.1,
            patience=3,
            lora_r=16,
            lora_alpha=32,
            device=device,
            num_classes=3,
            max_len=64,
        )

        set_seed(config.seed)

        tokenizer = AutoTokenizer.from_pretrained(config.model)
        print(
            f"""
    Tokenizer loaded:
    - Name or path: {tokenizer.name_or_path}
    - Vocab size: {tokenizer.vocab_size}
    - Max length: {tokenizer.model_max_length}
    """
        )

        train_df, val_df, test_df = load_and_split_data()
        train_loader = create_data_loader(
            train_df, tokenizer, config.max_len, config.batch_size
        )
        val_loader = create_data_loader(
            val_df, tokenizer, config.max_len, config.batch_size
        )
        test_loader = create_data_loader(
            test_df, tokenizer, config.max_len, config.batch_size
        )

        model = load_blank_model(config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        total_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device)

        history = defaultdict(list)
        best_f1 = 0
        epochs_without_improvement = 0

        for epoch in range(config.epochs):
            print(f"Epoch {epoch + 1}/{config.epochs}")
            print("-" * 10)

            train_f1, train_loss = train_epoch(
                model, train_loader, loss_fn, optimizer, device, scheduler
            )
            print(f"Train loss {train_loss:.4f} L-score {train_f1:.4f}")

            val_f1, val_loss = eval_model(model, val_loader, loss_fn, device)
            print(f"Val   loss {val_loss:.4f} L-score {val_f1:.4f}\n")

            history["train_f1"].append(train_f1)
            history["train_loss"].append(train_loss)
            history["val_f1"].append(val_f1)
            history["val_loss"].append(val_loss)

            if val_f1 > best_f1:
                torch.save(
                    model.state_dict(), f"{model_names[i]}.bin"
                )
                print(
                    f"Validation L-score improved from {best_f1:.4f} to {val_f1:.4f}. Model saved."
                )
                best_f1 = val_f1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement in L-score for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= config.patience:
                    print(
                        f"Early stopping triggered after {config.patience} epochs without improvement."
                    )
                    break


if __name__ == "__main__":
    main()
