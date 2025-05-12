import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_label_distribution(df, label_col='labels', title='Label Distribution'):
    """
    Plots and prints the distribution of labels in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the label column.
        label_col (str): The name of the label column.
        title (str): The title for the plot.
    """
    label_counts = df[label_col].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Print counts and percentages
    print(label_counts)
    print("\nClass distribution (percent):")
    print((label_counts / label_counts.sum() * 100).round(2))

def plot_tokens_sequence_lengths(df, tokenizer):
    """
    Plot the distribution of the number of tokens in the sequences.
    """
    token_lens = []

    for txt in df.text:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))

    sns.histplot(token_lens, bins=100, kde=True)
    plt.xlim([0, 256])
    plt.xlabel('Token count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths')
    plt.show()

def plot_training_history(history):
    """
    Plots training and validation f1-score from the training history.

    Parameters:
    - history (dict): A dictionary with keys 'train_acc' and 'val_acc', each containing a list of torch Tensors.
    """
    train_acc = history['train_acc'] # [elt.detach().cpu().numpy() for elt in history['train_acc']]
    val_acc = history['val_acc'] # [elt.detach().cpu().numpy() for elt in history['val_acc']]

    plt.plot(train_acc, label='Train f1-score')
    plt.plot(val_acc, label='Validation f1-score')

    plt.title('Training History')
    plt.ylabel('f1-score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn heatmap.

    Args:
        y_true (List[int] or np.array): Ground truth labels.
        y_pred (List[int] or np.array): Predicted labels.
        class_names (List[str]): Names of the classes (for labeling axes).
    """
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# https://medium.com/data-science/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    total_num_samples = len(dl.dataset)
    idxs_seen = []
    class_batch_counts = {i: [] for i in range(3)}  # For 3 classes

    for i, batch in enumerate(dl):
        targets = batch["targets"]
        class_ids, class_counts = targets.unique(return_counts=True)
        counts_dict = dict(zip(class_ids.tolist(), class_counts.tolist()))

        for class_id in range(3):
            class_batch_counts[class_id].append(counts_dict.get(class_id, 0))

        # Optional: track how many unique examples seen
        idxs_seen.extend(list(range(i * dl.batch_size, (i + 1) * dl.batch_size)))

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 6))
        ind = np.arange(len(class_batch_counts[0]))
        width = 0.25

        for i in range(3):
            ax.bar(
                ind + i * width,
                class_batch_counts[i],
                width,
                label=id_to_label[i] if id_to_label else f"Class {i}"
            )

        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of samples in batch", fontsize=12)
        ax.set_title("Class distribution per batch")
        ax.set_xticks(ind + width, labels=[str(i) for i in ind])
        ax.legend()
        plt.show()

        print("=============")
        print(f"Unique samples seen (approx): {len(set(idxs_seen))}/{total_num_samples}")
        for i in range(3):
            avg = np.array(class_batch_counts[i]).mean()
            print(f"Avg samples per batch for class {id_to_label[i] if id_to_label else i}: {avg:.2f}")
        print("=============")