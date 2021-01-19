import datetime
import json
import os

import pandas as pd
import torch
import gensim.downloader as gloader
from IPython.display import display, HTML


def get_nearest_answers(labels, preds, device="cpu"):
    """
    Given ground truths and predictions, return
    the nearest ground truth for each prediction
    """
    # Ensure to work with PyTorch tensors
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=device)
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, device=device)

    distances = torch.where(
        labels != -100,
        torch.abs(preds.unsqueeze(1).repeat(1, labels.shape[1], 1) - labels),
        -1,
    )
    summed_distances = torch.sum(distances, dim=2, keepdims=True)
    summed_distances[summed_distances < 0] = torch.max(summed_distances) + 1
    min_values, _ = torch.min(summed_distances, dim=1, keepdims=True)
    mask = (summed_distances == min_values).repeat(1, 1, 2)
    return labels[mask].reshape(preds.shape)


def show_df_row(df, index):
    """
    Show the given DataFrame row in a way
    that's enjoyable on Jupyter
    """
    row = df.iloc[index]
    display(HTML(pd.DataFrame([row]).to_html()))


def get_run_name():
    """
    Return a unique wandb run name
    """
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


def load_embedding_model(model_name, embedding_dimension=50):
    """
    Loads a pre-trained word embedding model via gensim library
    """
    model = f"{model_name}-{embedding_dimension}"
    try:
        return gloader.load(model)
    except Exception as e:
        print("Invalid embedding model name.")
        raise e


def save_answers(path, answers):
    """
    Save the given answer dict (in the format question-id: sub-context)
    to the given path, as a .json file
    """
    # fmt: off
    assert os.path.splitext(path)[1] == ".json", (
        "Answers can only be saved to a .json file"
    )
    # fmt: on
    with open(path, "w") as f:
        json.dump(answers, f)
