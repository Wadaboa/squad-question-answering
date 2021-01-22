import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import SquadTokenizer


class SquadDataset:
    """
    SQuAD question answering raw dataset wrapper
    """

    JSON_RECORD_PATH = ["data", "paragraphs", "qas", "answers"]

    def __init__(
        self, train_set_path=None, test_set_path=None, subset=1.0,
    ):
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

        self.raw_train_df = None
        if self.train_set_path is not None:
            assert os.path.exists(
                self.train_set_path
            ), "Missing SQuAD training set .json file"
            self.train_df_path = f"{os.path.splitext(self.train_set_path)[0]}.pkl"
            self.raw_train_df = self._load_dataset(
                self.train_set_path, self.train_df_path
            )
            if subset < 1.0:
                self.raw_train_df = self._get_portion(self.raw_train_df, subset)

        self.raw_test_df = None
        if self.test_set_path:
            assert os.path.exists(
                self.test_set_path
            ), "Missing SQuAD testing set .json file"
            self.test_df_path = f"{os.path.splitext(self.test_set_path)[0]}.pkl"
            self.raw_test_df = self._load_dataset(self.test_set_path, self.test_df_path)
            self.test_has_labels = "answer" in self.raw_test_df.columns
            if subset < 1.0:
                self.raw_test_df = self._get_portion(self.raw_test_df, subset)

    def _add_end_index(self, df):
        """
        Function that takes as input a partially-built SQuAD DataFrame 
        (with at least ['context', 'answer', 'answer_start'] columns)
        and returns the same DataFrame with the new column 'answer_end',
        that consists of last answer character index
        """
        ans_end = []
        for index, row in df.iterrows():
            t = row.answer
            s = row.answer_start
            ans_end.append(s + len(t))
        df["answer_end"] = ans_end
        return df

    def _load_dataset(self, dataset_path, dataframe_path):
        """
        Loads the SQuAD dataset into a Pandas DataFrame, 
        starting from a specifically-formatted JSON
        """
        # Load the DataFrame, if it was already pickled before
        if os.path.exists(dataframe_path):
            try:
                return pd.read_pickle(dataframe_path)
            except ValueError:
                pass

        # Check if the dataset has labels or not
        json_file = json.loads(open(dataset_path).read())
        if (
            len(
                pd.json_normalize(json_file, self.JSON_RECORD_PATH[:-1]).loc[
                    0, "answers"
                ]
            )
            > 0
        ):
            df = self._load_dataset_with_labels(json_file)
        else:
            df = self._load_dataset_no_labels(json_file)

        # Reset the index to [0, N]
        df = df.reset_index(drop=True)

        # Save the dataframe to a pickle file
        df.to_pickle(dataframe_path)

        return df

    def _load_dataset_with_labels(self, json_file):
        """
        Load a SQUaD dataset that has labels
        """
        # Flatten JSON
        df = pd.json_normalize(
            json_file, self.JSON_RECORD_PATH, meta=[["data", "title"]]
        )
        df_questions = pd.json_normalize(
            json_file, self.JSON_RECORD_PATH[:-1], meta=[["data", "title"]]
        )
        df_contexts = pd.json_normalize(
            json_file, self.JSON_RECORD_PATH[:-2], meta=[["data", "title"]]
        )

        # Build the flattened Pandas DataFrame
        contexts = np.repeat(df_contexts["context"].values, df_contexts.qas.str.len())
        contexts = np.repeat(contexts, df_questions["answers"].str.len())
        df["context"] = contexts
        df["question_id"] = np.repeat(
            df_questions["id"].values, df_questions["answers"].str.len()
        )
        df["question"] = np.repeat(
            df_questions["question"].values, df_questions["answers"].str.len()
        )
        df["context_id"] = df["context"].factorize()[0]

        # Rename columns
        df.rename(columns={"data.title": "title", "text": "answer"}, inplace=True)

        # Add end index for answers
        df = self._add_end_index(df)
        
        # Remove duplicated answers
        df = df.drop_duplicates()

        return df

    def _load_dataset_no_labels(self, json_file):
        """
        Load a SQUaD dataset that has no labels
        """
        # Flatten JSON
        df_questions = pd.json_normalize(
            json_file, self.JSON_RECORD_PATH[:-1], meta=[["data", "title"]]
        )
        df_contexts = pd.json_normalize(
            json_file, self.JSON_RECORD_PATH[:-2], meta=[["data", "title"]]
        )

        # Build the flattened Pandas DataFrame
        df_questions["context"] = np.repeat(
            df_contexts["context"].values, df_contexts.qas.str.len()
        )
        df_questions["context_id"] = df_questions["context"].factorize()[0]

        # Rename columns
        df_questions.rename(
            columns={"data.title": "title", "id": "question_id"}, inplace=True
        )
        
        if "answers" in df_questions.columns:
            df_questions = df_questions.drop("answers", axis="columns")

        return df_questions

    def _get_portion(self, df, subset=1.0):
        """
        Returns a random subset of the whole dataframe
        """
        amount = int(df.shape[0] * subset)
        random_indexes = np.random.choice(
            np.arange(df.shape[0]), size=amount, replace=False
        )
        return df.iloc[random_indexes].reset_index(drop=True)


class SquadDataManager:
    """
    SQuAD question answering dataset and tokenizer handler
    """

    def __init__(self, dataset, tokenizer, val_split=0.2, device="cpu"):
        assert isinstance(dataset, SquadDataset)
        assert isinstance(tokenizer, SquadTokenizer)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.val_split = val_split
        self.device = device

        # Preprocess DataFrames and perform train/val split
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if self.dataset.raw_train_df is not None:
            self.train_df, self.val_df = self._train_val_split(
                self.preprocess(self.dataset.raw_train_df.copy()), self.val_split
            )
            self.train_dataset = SquadTorchDataset(self.train_df)
            self.val_dataset = SquadTorchDataset(self.val_df)
        if self.dataset.raw_test_df is not None:
            self.test_df = self.dataset.raw_test_df.copy()
            if self.dataset.test_has_labels:
                self.test_df = self.preprocess(self.test_df)
            self.test_dataset = SquadTorchDataset(self.test_df)

    def preprocess(self, df):
        """
        Performs all the preprocessing steps with the given DataFrame
        """
        df = self._remove_lost_answers(df)
        df = self._group_answers(df)
        return df

    def _remove_lost_answers(self, df):
        """
        Remove rows that contain incorrect answers, (either 
        because of the the tokenizer's truncation or because
        they were not correct in the first place)
        """
        tokenized_contexts = self.tokenizer.tokenize(
            df["context"].tolist(), "context", special=False
        )
        lost_truncated, lost_dirty = self._lost_answers_indexes(df, tokenized_contexts)
        to_remove = lost_truncated + lost_dirty
        clean_df = df.drop(to_remove)
        assert len(clean_df) == len(df) - len(to_remove), (
            f"Before {len(df)}, " f"after {len(clean_df)}, " f"removed {len(to_remove)}"
        )
        return clean_df

    def _lost_answers_indexes(self, df, tokenized_contexts):
        """
        Returns two lists, where the first one contains answers that would be
        lost due to tokenization, while the second one contains wrong answers
        """
        whole_answers_start = df["answer_start"].tolist()
        whole_answers_end = df["answer_end"].tolist()
        lost_dirty, lost_truncated = [], []
        for i, (c, s, e) in enumerate(
            zip(tokenized_contexts, whole_answers_start, whole_answers_end)
        ):
            mask = (
                torch.tensor(c.attention_mask, device=self.device).bool()
                & ~torch.tensor(c.special_tokens_mask, device=self.device).bool()
            )
            offsets = torch.tensor(c.offsets, device=self.device)[mask]
            start_index = torch.nonzero(offsets[:, 0] == s)
            end_index = torch.nonzero(offsets[:, 1] == e)
            if len(start_index) == 0 or len(end_index) == 0:
                if s > offsets[-1, 0] or e > offsets[-1, 1]:
                    lost_truncated.append(i)
                else:
                    lost_dirty.append(i)

        return lost_truncated, lost_dirty

    def _group_answers(self, df):
        """
        Group answers to the same question into a single row
        """
        return (
            df.groupby(["question_id", "question", "title", "context_id", "context"])
            .agg({"answer": list, "answer_start": list, "answer_end": list})
            .reset_index()
        )

    def _train_val_split(self, df, val_split):
        """
        Perform train/validation splits, with the specified ratio
        """
        # Compute the number of validation examples
        val_size = round(df.shape[0] * val_split)

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

        if "answer" not in self.df.columns:
            return index, question, context

        answer_start = self.df.loc[index, "answer_start"]
        answer_end = self.df.loc[index, "answer_end"]
        return index, question, context, answer_start, answer_end
