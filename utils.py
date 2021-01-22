import datetime
import json
import os
import re
import string
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers
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

    # Compute the euclidean distance between each pair of
    # related (label, prediction), while ignoring pad values
    distances = torch.where(
        labels != -100,
        torch.pow(preds.unsqueeze(1).repeat(1, labels.shape[1], 1) - labels, 2),
        -100,
    )
    summed_distances = torch.sqrt(torch.sum(distances, dim=2, keepdims=True).float())
    summed_distances[torch.isnan(summed_distances)] = torch.tensor(
        np.inf, device=device
    )

    # Consider the closest labels to the given predictions
    _, min_indexes = torch.min(summed_distances, dim=1, keepdims=True)
    mask = torch.zeros_like(summed_distances, device=device).bool()
    mask = mask.scatter(
        1, min_indexes, torch.ones_like(min_indexes, device=device).bool()
    ).repeat(1, 1, 2)
    return labels[mask].reshape(preds.shape)


def show_df_row(df, index):
    """
    Show the given DataFrame row in a way
    that's enjoyable on Jupyter
    """
    row = df.iloc[index]
    display(HTML(pd.DataFrame([row]).to_html()))


def get_run_name(fmt="%Y-%m-%d--%H-%M-%S"):
    """
    Return a unique wandb run name
    """
    return datetime.datetime.now().strftime(fmt)


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
        # (if the mean is already present, use a random vector)
        assert unk_token not in embedding_model, f"{unk_token} key already present"
        unk = np.mean(embedding_model.vectors, axis=0)
        if unk in embedding_model.vectors:
            mins = np.min(embedding_model.vectors, axis=0)
            maxs = np.max(embedding_model.vectors, axis=0)
            unk = (maxs - mins) * np.random.rand(embedding_dimension) + mins
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
    """
    Return a CUDA device, if available, or a standard CPU device otherwise
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_default_trainer_args(
    logging_dir="./runs",
    logging_steps=5,
    metric_for_best_model="f1",
    label_names=["answers"],
    seed=1,
):
    """
    Return default parameters for the SQUaD trainer
    """
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


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace
    """

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def from_words_to_text(df, spans, indexes):
    """
    Compute answers (taking spans from original contexts)
    """
    answers_dict = dict()
    for span, index in zip(spans, indexes):
        answer = df.loc[index, "context"][span[0] : span[1] + 1]
        answers_dict[df.loc[index, "question_id"]] = normalize_answer(answer)
    return answers_dict
