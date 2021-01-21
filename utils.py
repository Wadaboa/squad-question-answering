import datetime
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers
import gensim.downloader as gloader
from IPython.display import display, HTML


def get_nearest_answers(labels, preds, eps=1e-3, device="cpu"):
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
        torch.pow(preds.unsqueeze(1).repeat(1, labels.shape[1], 1) - labels, 2),
        -100,
    )
    summed_distances = torch.sqrt(torch.sum(distances, dim=2, keepdims=True).float())
    summed_distances[torch.isnan(summed_distances)] = torch.tensor(
        np.inf, device=device
    )
    summed_distances += torch.rand(summed_distances.shape, device=device) * eps
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


def load_embedding_model(
    model_name, embedding_dimension=50, unk_token="[UNK]", pad_token="[PAD]"
):
    """
    Loads a pre-trained word embedding model via gensim library
    """
    model = f"{model_name}-{embedding_dimension}"
    try:
        embedding_model = gloader.load(model)

        # Build the unknown vector as the mean of all vectors
        assert unk_token not in embedding_model, f"{unk_token} key already present"
        unk = np.mean(embedding_model.vectors, axis=0)
        assert unk not in embedding_model.vectors, f"{unk_token} value already present"
        embedding_model.add(unk_token, unk)

        # Build the pad vector as a zero vector
        assert pad_token not in embedding_model, f"{pad_token} key already present"
        pad = np.zeros((1, embedding_model.vectors.shape[1]))
        assert pad not in embedding_model.vectors, f"{pad_token} value already present"
        embedding_model.add(pad_token, pad)

        # Extract a mapping from keys to indexes
        vocab = dict(
            zip(embedding_model.index2word, range(len(embedding_model.index2word)))
        )

        return embedding_model, vocab
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


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_default_trainer_args(
    logging_dir="./runs",
    logging_steps=5,
    metric_for_best_model="f1",
    label_names=["answers"],
    seed=1,
):
    return partial(
        transformers.TrainingArguments,
        logging_dir="./runs",
        logging_first_step=True,
        logging_steps=5,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        label_names=["answers"],
        seed=seed,
    )
