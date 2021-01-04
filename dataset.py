import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def static_padding(sequences, shape, padding):
    """
    Given a sequence of tensors with different lenghts and a fixed shape,
    fit the tensors in a single one by padding to the given shape
    """
    out_tensor = torch.empty(*shape, dtype=torch.long).fill_(padding)
    for i, elem in enumerate(sequences):
        tensor = torch.tensor(elem)
        lenght = tensor.size(0)
        out_tensor[i, :lenght, ...] = tensor
    return out_tensor


def pad_batch(batch, padding_index, max_question_tokens, max_context_tokens):
    """
    This function expects to receive a list of tuples (i.e. a batch),
    s.t. each tuple is made of (question, context, answer_start, answer_end)
    elements, and returns the same sequences padded with the padding token
    """
    (questions, contexts, answers_start, answers_end) = zip(*batch)
    questions_lenghts = torch.tensor([len(x) for x in questions], dtype=torch.long)
    contexts_lenghts = torch.tensor([len(y) for y in contexts], dtype=torch.long)
    padded_questions = static_padding(
        questions, (len(batch), max_question_tokens), padding_index
    )
    padded_contexts = static_padding(
        contexts, (len(batch), max_context_tokens), padding_index
    )
    answers_start_tensor = torch.tensor(answers_start, dtype=torch.long)
    answers_end_tensor = torch.tensor(answers_end, dtype=torch.long)
    return (
        padded_questions,
        padded_contexts,
        answers_start_tensor,
        answers_end_tensor,
        questions_lenghts,
        contexts_lenghts,
    )


class SquadDataset:
    """
    SQuAD question answering dataset wrapper
    """

    NAME = "squad"
    COLUMNS = [
        "index",
        "question",
        "title",
        "context_id",
        "context",
        "text",
        "answer_start",
        "answer_end",
    ]
    JSON_RECORD_PATH = ["data", "paragraphs", "qas", "answers"]

    def __init__(
        self,
        max_question_tokens,
        max_context_tokens,
        df_percentage=1.0,
        question_preprocessor=None,
        context_preprocessor=None,
        val_split=0.2,
        batch_size=10,
    ):
        # Save instance variables
        self.df_percentage = df_percentage
        self.val_split = val_split
        self.question_preprocessor = question_preprocessor
        self.context_preprocessor = context_preprocessor

        # Create directories
        self.dataset_folder = os.path.join(os.getcwd(), "data", "training")
        self.training_set_path = os.path.join(self.dataset_folder, "training_set.json")
        self.dataframe_path = os.path.join(self.dataset_folder, f"{self.NAME}_df.pkl")
        assert os.path.exists(
            self.training_set_path
        ), "Missing SQuAD training set .json file"

        # Prepare the dataset and load the dataframe
        if not os.path.exists(self.dataframe_path):
            self.dataframe = self._load_dataset()
        else:
            self.dataframe = pd.read_pickle(self.dataframe_path)

        # Store a random subset of the entire DataFrame
        if self.df_percentage < 1.0:
            self.dataframe = self._get_portion(self.dataframe)

        # Preprocess the dataframe
        if (
            self.question_preprocessor is not None
            or self.context_preprocessor is not None
        ):
            self.dataframe = self._preprocess(self.dataframe)

        # Split the DataFrame into train/validation
        self.train_df, self.val_df = self._train_val_split(self.dataframe)

        # Save PyTorch Dataset instances
        self.train_dataset = SquadTorchDataset(self.train_df)
        self.val_dataset = SquadTorchDataset(self.val_df)

        # Declare PyTorch DataLoader instances
        collate_fn = partial(
            pad_batch,
            padding_index=0,
            max_question_tokens=max_question_tokens,
            max_context_tokens=max_context_tokens,
        )
        default_dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.train_dataloader = default_dataloader(self.train_dataset)
        self.val_dataloader = default_dataloader(self.val_dataset)

    def _add_end_index(self, df):
        """
        Function that takes as input a partially-built SQuAD DataFrame 
        (with at least ['context', 'text', 'answer_start'] columns)
        and returns the same DataFrame with the new column 'answer_end',
        that consists of last answer character index
        """
        ans_end = []
        for index, row in df.iterrows():
            t = row.text
            s = row.answer_start
            ans_end.append(s + len(t))
        df["answer_end"] = ans_end
        return df

    def _load_dataset(self):
        """
        Loads the SQuAD dataset into a Pandas DataFrame, starting from the training set JSON
        """
        # Read JSON file
        file = json.loads(open(self.training_set_path).read())

        # Flatten JSON
        js = pd.io.json.json_normalize(
            file, self.JSON_RECORD_PATH, meta=[["data", "title"]]
        )
        m = pd.io.json.json_normalize(
            file, self.JSON_RECORD_PATH[:-1], meta=[["data", "title"]]
        )
        r = pd.io.json.json_normalize(
            file, self.JSON_RECORD_PATH[:-2], meta=[["data", "title"]]
        )

        # Build the flattened Pandas DataFrame
        idx = np.repeat(r["context"].values, r.qas.str.len())
        ndx = np.repeat(m["id"].values, m["answers"].str.len())
        m["context"] = idx
        js["q_idx"] = ndx
        df = pd.concat(
            [m[["id", "question", "context"]].set_index("id"), js.set_index("q_idx")],
            axis=1,
            sort=False,
        ).reset_index()
        df["context_id"] = df["context"].factorize()[0]
        df.rename(columns={"data.title": "title"}, inplace=True)

        # Add end index for answers in DataFrame
        df = self._add_end_index(df)

        # Order columns
        df = df[self.COLUMNS[-1]]
        return df

    def _preprocess(self, df):
        """
        Apply the right preprocessing functions to the
        question/context columns
        """
        if self.context_preprocessor is not None:
            df["context"] = df["context"].apply(self.context_preprocessor)
            df["text"] = df["text"].apply(self.context_preprocessor)
        if self.question_preprocessor is not None:
            df["question"] = df["question"].apply(self.context_preprocessor)
        return df

    def _train_val_split(self, df):
        """
        Perform train/validation splits, with the specified ratio
        """
        # Compute the number of validation examples
        val_size = round(self.dataframe.shape[0] * self.val_split)

        # Compute validation examples by keeping all questions related
        # to the same context within the same split
        val_actual_size = 0
        val_keys = []
        for t, n in df["title"].value_counts().to_dict().items():
            if val_actual_size + n > val_size:
                break
            val_keys.append(t)
            val_actual_size += n

        # Build the train and validation DataFrames
        train_df = df[~df["title"].isin(val_keys)].reset_index(drop=True)
        val_df = df[df["title"].isin(val_keys)].reset_index(drop=True)
        return train_df, val_df

    def _get_portion(self, df):
        """
        Returns a random subset of the whole dataframe
        """
        amount = df.shape[0] * self.df_percentage
        random_indexes = np.random.choice(
            np.arange(df.shape[0]), size=amount, replace=False
        )
        return df.iloc[random_indexes]


class SquadTorchDataset(Dataset):
    """
    SQuAD question answering PyTorch Dataset subclass
    """

    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int)
        question = self.df.loc[index, "question"]
        context = self.df.loc[index, "context"]
        answer_start = self.df.loc[index, "answer_start"]
        answer_end = self.df.loc[index, "answer_end"]
        return question, context, answer_start, answer_end
