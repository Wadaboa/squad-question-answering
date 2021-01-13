import json
import os
import uuid
from functools import partial
from operator import attrgetter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class SquadDataset:
    """
    SQuAD question answering raw dataset wrapper
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
        self, subset=1.0,
    ):
        assert os.path.exists(
            self.TRAIN_SET_PATH
        ), "Missing SQuAD training set .json file"
        assert os.path.exists(
            self.TEST_SET_PATH
        ), "Missing SQuAD testing set .json file"

        # Prepare the dataset and load the dataframe
        self.raw_train_df = self._load_dataset(self.TRAIN_SET_PATH, self.TRAIN_DF_PATH)
        self.raw_test_df = self._load_dataset(self.TEST_SET_PATH, self.TEST_DF_PATH)
        self.subset = subset
        if self.subset < 1.0:
            self.raw_train_df = self._get_portion(self.raw_train_df, subset)

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

        # Add end index for answers and remove duplicated ones
        df = self._add_end_index(df)
        df = self._remove_duplicated_answers(df)
        
        # Order columns and reset index
        df = df[self.COLUMNS]
        df = df.reset_index(drop=True)
        
        # Save the dataframe to a pickle file
        df.to_pickle(dataframe_path)

        return df

    def _get_portion(self, df, subset=1.0):
        """
        Returns a random subset of the whole dataframe
        """
        amount = df.shape[0] * subset
        random_indexes = np.random.choice(
            np.arange(df.shape[0]), size=amount, replace=False
        )
        return df.iloc[random_indexes]


class SquadDataManager:
    """
    SQuAD question answering dataset and tokenizer handler
    """

    def __init__(self, dataset, tokenizer, val_split=0.2):
        assert isinstance(dataset, SquadDataset)
        assert isinstance(tokenizer, SquadTokenizer)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.val_split = val_split

        # Preprocess DataFrames and perform train/val split
        self.train_df, self.val_df = self._train_val_split(
            self.preprocess(self.dataset.raw_train_df.copy()), self.val_split
        )
        self.test_df = self.preprocess(self.dataset.raw_test_df.copy())

        # Save PyTorch Dataset instances
        self.train_dataset = SquadTorchDataset(self.train_df)
        self.val_dataset = SquadTorchDataset(self.val_df)
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
            offsets = torch.tensor(c.offsets)[torch.tensor(c.attention_mask).bool()]
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


class SquadTokenizer:
    """
    Abstract tokenizer and data collator interface
    for the SQuAD dataset
    """

    def tokenize(self, inputs, entity, special=False):
        tokenizer = self.select_tokenizer(entity)
        tokenizer_padding = tokenizer.padding
        if not special:
            tokenizer.no_padding()
        outputs = tokenizer.encode_batch(inputs, add_special_tokens=special)
        tokenizer.enable_padding(**tokenizer_padding)
        return outputs

    def find_tokenized_answer_indexes(self, offsets, query, start):
        assert isinstance(start, bool)
        index = torch.nonzero(offsets[:, int(start)] == query)
        assert len(index) in (0, 1)
        return index.item() if len(index) > 0 else -1

    def select_tokenizer(self, entity):
        raise NotImplementedError()

    def __call__(self, inputs):
        raise NotImplementedError()


class StandardSquadTokenizer(SquadTokenizer):
    """
    Standard SQuAD tokenizer and data collator,
    based on the usage of two different tokenizer
    (one for questions and one for contexts)
    """

    ENCODING_ATTR = [
        "ids",
        "type_ids",
        "tokens",
        "offsets",
        "attention_mask",
        "special_tokens_mask",
        "overflowing",
    ]
    ENCODING_ATTR_ID = dict(enumerate(ENCODING_ATTR))
    ATTRGETTER = attrgetter(*ENCODING_ATTR)

    def __init__(self, question_tokenizer, context_tokenizer):
        super().__init__()
        assert isinstance(question_tokenizer, Tokenizer)
        assert isinstance(context_tokenizer, Tokenizer)
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def select_tokenizer(self, entity):
        assert entity in ("question", "context")
        return (
            self.context_tokenizer if entity == "context" else self.question_tokenizer
        )

    def __call__(self, inputs):
        (questions, contexts, answers_start, answers_end) = zip(*inputs)
        tokenized_questions = self.tokenize(questions, "question", special=True)
        tokenized_contexts = self.tokenize(questions, "context", special=True)
        qattr = list(zip(*[self.ATTRGETTER(e) for e in tokenized_questions]))
        qattr = list(zip(*[self.ATTRGETTER(e) for e in tokenized_contexts]))

        # Create the batch dictionary with encoding info
        batch = {
            "question_ids": torch.tensor(qattr[self.ENCODING_ATTR_ID["ids"]]),
            "question_type_ids": torch.tensor(qattr[self.ENCODING_ATTR_ID["type_ids"]]),
            "question_attention_mask": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["attention_mask"]]
            ),
            "question_special_tokens_mask": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["special_tokens_mask"]]
            ),
            "context_ids": torch.tensor(cattr[self.ENCODING_ATTR_ID["ids"]]),
            "context_type_ids": torch.tensor(cattr[self.ENCODING_ATTR_ID["type_ids"]]),
            "context_attention_mask": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["attention_mask"]]
            ),
            "context_special_tokens_mask": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["special_tokens_mask"]]
            ),
            "context_offsets": torch.tensor(cattr[self.ENCODING_ATTR_ID["offsets"]]),
        }

        # Add custom info to the batch dict
        masked_offsets = batch["context_offsets"][batch["context_attention_mask"]]
        batch["answer_start"] = self.find_tokenized_answer_indexes(
            masked_offsets, answers_start, True
        )
        batch["answer_end"] = self.find_tokenized_answer_indexes(
            masked_offsets, answers_end, False
        )
        batch["question_lenghts"] = torch.count_nonzero(
            ~batch["question_special_tokens_mask"]
        )
        batch["context_lenghts"] = torch.count_nonzero(
            ~batch["context_special_tokens_mask"]
        )

        return batch


class BertSquadTokenizer(SquadTokenizer):
    """
    Tokenizer and data collator to be used with
    a BERT model
    """

    def __init__(tokenizer):
        super().__init__()
        assert isinstance(tokenizer, Tokenizer)
        self.tokenizer = tokenizer

    def select_tokenizer(self, entity):
        return self.tokenizer


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
