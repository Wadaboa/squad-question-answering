import json
import os
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SquadDataset:
    """
    SQuAD question answering dataset wrapper
    """

    NAME = "squad"
    COLUMNS = [
        "question_id",
        "question",
        "title",
        "context_id",
        "context",
        "answer",
        "answer_start",
        "answer_end",
    ]
    JSON_RECORD_PATH = ["data", "paragraphs", "qas", "answers"]

    DATA_FOLDER = os.path.join(os.getcwd(), "data")
    TRAIN_DATA_FOLDER = os.path.join(DATA_FOLDER, "training")
    TRAIN_SET_PATH = os.path.join(TRAIN_DATA_FOLDER, "training_set.json")
    TRAIN_DF_PATH = os.path.join(TRAIN_DATA_FOLDER, f"{NAME}_train_df.pkl")
    TEST_DATA_FOLDER = os.path.join(DATA_FOLDER, "testing")
    TEST_SET_PATH = os.path.join(TEST_DATA_FOLDER, "test_set.json")
    TEST_DF_PATH = os.path.join(TEST_DATA_FOLDER, f"{NAME}_test_df.pkl")

    def __init__(
        self, df_percentage=1.0, val_split=0.2,
    ):
        # Save instance variables
        self.df_percentage = df_percentage
        self.val_split = val_split

        assert os.path.exists(
            self.TRAIN_SET_PATH
        ), "Missing SQuAD training set .json file"
        assert os.path.exists(
            self.TEST_SET_PATH
        ), "Missing SQuAD testing set .json file"

        # Prepare the dataset and load the dataframe
        train_df = self._load_dataset(self.TRAIN_SET_PATH, self.TRAIN_DF_PATH)
        test_df = self._load_dataset(self.TEST_SET_PATH, self.TEST_DF_PATH)

        # Apply train/val split, store the DataFrame and the
        # corresponding PyTorch Dataset
        self.update_df(train_df)
        self.update_df(test_df, test=True)

    def update_df(self, df, test=False):
        """
        Update the DataFrame and PyTorch Dataset with the given one
        """
        if test:
            self.test_df = df
            self.test_dataset = SquadTorchDataset(self.test_df)
        else:
            # Update the DataFrame
            self.dataframe = df

            # Store a random subset of the entire DataFrame
            if self.df_percentage < 1.0:
                self.dataframe = self._get_portion(self.dataframe)

            # Split the DataFrame into train/validation
            self.train_df, self.val_df = self._train_val_split(self.dataframe)

            # Save PyTorch Dataset instances
            self.train_dataset = SquadTorchDataset(self.train_df)
            self.val_dataset = SquadTorchDataset(self.val_df)

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
    
    def _remove_duplicated_answers(self, df):
        """
        Remove duplicated rows from the given DataFrame
        """
        return df.drop_duplicates()

    def _load_dataset(self, dataset_path, dataframe_path):
        """
        Loads the SQuAD dataset into a Pandas DataFrame, starting from the training set JSON
        """
        if os.path.exists(dataframe_path):
            return pd.read_pickle(dataframe_path)

        # Read JSON file
        file = json.loads(open(dataset_path).read())

        # Flatten JSON
        df = pd.json_normalize(file, self.JSON_RECORD_PATH, meta=[["data", "title"]])
        df_questions = pd.json_normalize(
            file, self.JSON_RECORD_PATH[:-1], meta=[["data", "title"]]
        )
        df_contexts = pd.json_normalize(
            file, self.JSON_RECORD_PATH[:-2], meta=[["data", "title"]]
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

        # Add end index for answers in DataFrame
        df = self._add_end_index(df)

        # Order columns
        df = df[self.COLUMNS]
        
        # Remove duplicate answers
        df = self._remove_duplicated_answers(df)
        
        # Save the dataframe to a pickle file
        df.to_pickle(dataframe_path)

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
