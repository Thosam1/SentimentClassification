from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
)

from config.config import RANDOM_SEED
from data_preprocessing.dataset_dataloader import create_data_loader
from utils.utils import L_score, get_device


@dataclass
class ConfigBase:
    batch_size: int
    model: str
    seed: int
    lr: float
    dropout: float
    attention_dropout: float
    device: str
    num_classes: int
    max_len: int

@dataclass
class ConfigExtended(ConfigBase):
    epochs: int
    patience: int

def load_config_base():
    return ConfigBase(
        batch_size=16,
        model="FacebookAI/roberta-large",
        seed=RANDOM_SEED,
        lr=3e-5,
        dropout=0.1,
        attention_dropout=0.1,
        device=get_device(),
        num_classes=3,
        max_len=64,
    )


# We will use this function later to reload the model from scratch
def load_blank_model(config):
    # Delete references to a previously loaded model
    if "optimizer" in globals():
        global optimizer
        del optimizer
    if "model" in globals():
        global model
        del model

    # Free up GPU memory
    torch.cuda.empty_cache()

    # Modify dropout parameters in the config
    model_config = AutoConfig.from_pretrained(config.model)
    model_config.hidden_dropout_prob = config.dropout
    model_config.attention_probs_dropout_prob = config.attention_dropout
    model_config.num_labels = config.num_classes

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, config=model_config
    )
    model.to(config.device)

    return model


def load_model_with_data_loader(config, model_name, model_path, test_df):
    config.model = model_name
    device = get_device()

    llm_tokenizer = AutoTokenizer.from_pretrained(config.model)
    llm_model = load_blank_model(config)

    # Load saved model state dict
    state_dict = torch.load(model_path, map_location=device)
    llm_model.load_state_dict(state_dict)

    llm_model.to(device)
    llm_model.eval()

    loss_fn = nn.CrossEntropyLoss().to(device)
    test_data_loader = create_data_loader(
        test_df, llm_tokenizer, config.max_len, config.batch_size
    )

    return llm_model, loss_fn, test_data_loader


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()

    losses = []
    all_preds = []
    all_targets = []

    loop = tqdm(data_loader, desc="Training", leave=False)

    for d in loop:
        logits, preds, targets = model_predictions(d, device, model)
        loss = loss_fn(logits, targets)

        losses.append(loss.item())

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update tqdm description with F1 for current batch (optional, coarse)
        batch_f1 = f1_score(all_targets, all_preds, average="macro")
        loop.set_postfix(loss=loss.item(), f1=round(batch_f1, 4))

    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_f1, np.mean(losses)


from sklearn.metrics import f1_score


@torch.inference_mode()
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()

    losses = []
    all_preds = []
    all_targets = []

    loop = tqdm(data_loader, desc="Evaluating", leave=False)

    for d in loop:
        logits, preds, targets = model_predictions(d, device, model)

        loss = loss_fn(logits, targets)

        losses.append(loss.item())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # Optional: show running F1 during evaluation
        # current_f1 = f1_score(all_targets, all_preds, average="macro")
        # loop.set_postfix(loss=loss.item(), f1=round(current_f1, 4))
        current_L_score = L_score(all_targets, all_preds)
        loop.set_postfix(loss=loss.item(), L_score=round(current_L_score, 4))

    # epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    epoch_L_score = L_score(all_targets, all_preds)
    return epoch_L_score, np.mean(losses)


@torch.inference_mode()
def get_predictions(model, data_loader, device):
    """
    Generate predictions, prediction probabilities, and true labels from a given model and data loader.

    Args:
        model (torch.nn.Module): Trained model for sequence classification.
        data_loader (DataLoader): DataLoader containing evaluation/test data.
        device (str): Device to run inference on (e.g., 'cuda' or 'cpu').

    Returns:
        review_texts (List[str]): Original input texts.
        predictions (torch.Tensor): Predicted class indices.
        prediction_probs (torch.Tensor): Softmax probabilities for each class.
        real_values (torch.Tensor): True labels.
    """
    model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    for d in data_loader:
        texts = d["review_text"]
        logits, preds, targets = model_predictions(d, device, model)

        probs = F.softmax(logits, dim=1)

        review_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(probs)
        real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values


def model_predictions(d, device, model):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    _, preds = torch.max(logits, dim=1)
    return logits, preds, targets


def predict_with_ensemble(
    preds_list: List[torch.Tensor],
    probs_list: Optional[List[torch.Tensor]] = None,
    strategy: Literal["majority", "softmax_avg", "weighted"] = "majority",
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Ensemble predictions using the specified strategy.

    Args:
        preds_list: List of predicted class tensors from different variations (e.g., base, var1, var2).
        probs_list: List of softmax probability tensors from each variation.
        strategy: Strategy to use: 'majority', 'softmax_avg', or 'weighted'.
        weights: Optional list of weights for 'weighted' softmax voting. Must match length of probs_list.

    Returns:
        torch.Tensor: Final predicted class for each example.
    """
    num_variants = len(preds_list)
    num_samples = preds_list[0].size(0)

    if strategy == "majority":
        preds_stacked = torch.stack(
            preds_list, dim=1
        )  # shape: (num_samples, num_variants)
        final_preds = []

        for i in range(num_samples):
            row = preds_stacked[i].tolist()
            count = Counter(row)
            most_common = count.most_common()

            if len(most_common) == 1 or most_common[0][1] > 1:
                final_preds.append(most_common[0][0])
            elif probs_list:
                avg_probs = torch.stack([probs[i] for probs in probs_list]).mean(dim=0)
                final_preds.append(torch.argmax(avg_probs).item())
            else:
                final_preds.append(row[0])  # fallback

        return torch.tensor(final_preds)

    elif strategy == "softmax_avg":
        assert (
            probs_list is not None
        ), "Softmax probabilities required for softmax_avg strategy"
        probs_stack = torch.stack(
            probs_list
        )  # shape: (num_variants, num_samples, num_classes)
        avg_probs = probs_stack.mean(dim=0)  # shape: (num_samples, num_classes)
        return torch.argmax(avg_probs, dim=1)

    elif strategy == "weighted":
        assert (
            probs_list is not None
        ), "Softmax probabilities required for weighted strategy"
        assert (
            weights is not None and len(weights) == num_variants
        ), "Weights must be same length as number of variants"

        weighted_probs = sum(w * p for w, p in zip(weights, probs_list))
        return torch.argmax(weighted_probs, dim=1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
