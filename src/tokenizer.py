from operator import attrgetter

import torch
from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Sequence, StripAccents, Lowercase, Strip
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers import BertWordPieceTokenizer


class SquadTokenizer:
    """
    Abstract tokenizer and data collator interface
    for the SQuAD dataset
    """

    ENCODING_ATTR = [
        "ids",
        "type_ids",
        "tokens",
        "offsets",
        "attention_mask",
        "special_tokens_mask",
        "overflowing",
        "word_ids",
    ]
    ENCODING_ATTR_ID = {k: i for i, k in enumerate(ENCODING_ATTR)}
    ATTRGETTER = attrgetter(*ENCODING_ATTR)

    def __init__(self, device="cpu"):
        self.device = device

    def tokenize(self, inputs, entity=None, special=False):
        """
        Tokenize the given input, using the specified tokenizer
        (optionally remove special tokens, like padding)
        """
        tokenizer = self.select_tokenizer(entity)
        tokenizer_padding = tokenizer.padding
        if not special:
            tokenizer.no_padding()
        outputs = tokenizer.encode_batch(inputs, add_special_tokens=special)
        tokenizer.enable_padding(**tokenizer_padding)
        return outputs

    def detokenize(self, inputs, entity=None, special=True):
        """
        De-tokenize the given tokenized input, into
        the original word sequence
        """
        tokenizer = self.select_tokenizer(entity)
        return tokenizer.decode_batch(inputs, skip_special_tokens=not special)

    def get_pad_token_id(self):
        """
        Return the ID of the token used for padding
        """
        tokenizer = self.select_tokenizer(entity="context")
        return tokenizer.padding["pad_id"]

    def find_tokenized_answer_indexes(self, offsets, starts, ends):
        """
        Given start and end word indices, return the corresponding
        indices in the tokenized space
        """
        batch_size = len(starts)
        max_answers = max([len(row) for row in starts])
        indexes = torch.full((batch_size, max_answers, 2), -100, device=self.device)
        for i, (start, end) in enumerate(zip(starts, ends)):
            for j, (s, e) in enumerate(zip(start, end)):
                start_index = torch.nonzero(offsets[i, :, 0] == s)
                end_index = torch.nonzero(offsets[i, :, 1] == e)
                if len(start_index) > 0 and len(end_index) > 0:
                    indexes[i, j, :] = torch.tensor(
                        [start_index[0], end_index[0]], device=self.device
                    )
        return indexes

    def find_subword_indexes(self, word_ids):
        """
        Given a list of word IDs, return two mask,
        one indicating the start of words and one
        indicating the end of words, which should 
        be used to ignore subwords
        """
        start_mask = torch.full(
            (len(word_ids), len(word_ids[0])), False, device=self.device
        )
        end_mask = torch.full_like(start_mask, False, device=self.device)

        for i, word_id in enumerate(word_ids):

            if word_id[0] != None:
                start_mask[i, 0] = True

            for j in range(1, len(word_id)):
                if word_id[j] != word_id[j - 1]:
                    if word_id[j] != None:
                        start_mask[i, j] = True
                    if word_id[j - 1] != None:
                        end_mask[i, j] = True

            if word_id[-1] != None:
                end_mask[i, -1] = True

        return start_mask, end_mask

    def select_tokenizer(self, entity=None):
        """
        Given an entity (None, "context" or "question"),
        return the corresponding tokenizer
        """
        raise NotImplementedError()

    def __call__(self, inputs):
        raise NotImplementedError()


class RecurrentSquadTokenizer(SquadTokenizer):
    """
    Tokenizer and data collator to be used with a 
    recurrent-based model, which uses two different tokenizers
    (one for questions and one for contexts)
    """

    def __init__(self, question_tokenizer, context_tokenizer, device="cpu"):
        super().__init__(device=device)
        assert isinstance(question_tokenizer, Tokenizer)
        assert isinstance(context_tokenizer, Tokenizer)
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def select_tokenizer(self, entity=None):
        """
        Return the "question" or "context" tokenizer,
        based on the given entity
        """
        assert entity in ("question", "context")
        return (
            self.context_tokenizer if entity == "context" else self.question_tokenizer
        )

    def __call__(self, inputs):
        zipped_inputs = tuple(zip(*inputs))
        if len(zipped_inputs) > 3:
            (indexes, questions, contexts, answers_start, answers_end) = zipped_inputs
            testing = False
        else:
            (indexes, questions, contexts) = zipped_inputs
            testing = True
        tokenized_questions = self.tokenize(questions, entity="question", special=True)
        tokenized_contexts = self.tokenize(contexts, entity="context", special=True)
        qattr = list(zip(*[self.ATTRGETTER(e) for e in tokenized_questions]))
        cattr = list(zip(*[self.ATTRGETTER(e) for e in tokenized_contexts]))

        # Create the batch dictionary with encoding info
        batch = {
            "question_ids": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["ids"]], device=self.device
            ),
            "question_type_ids": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["type_ids"]], device=self.device
            ),
            "question_attention_mask": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["attention_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "question_special_tokens_mask": torch.tensor(
                qattr[self.ENCODING_ATTR_ID["special_tokens_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "context_ids": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["ids"]], device=self.device
            ),
            "context_type_ids": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["type_ids"]], device=self.device
            ),
            "context_attention_mask": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["attention_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "context_special_tokens_mask": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["special_tokens_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "context_offsets": torch.tensor(
                cattr[self.ENCODING_ATTR_ID["offsets"]], device=self.device
            ),
            "indexes": torch.tensor(indexes, dtype=torch.long, device=self.device),
        }

        # Add custom info to the batch dict
        batch["context_offsets"] = torch.where(
            batch["context_attention_mask"].unsqueeze(-1).repeat(1, 1, 2),
            batch["context_offsets"],
            -100,
        )
        batch["question_lenghts"] = torch.count_nonzero(
            batch["question_attention_mask"], dim=1
        )
        batch["context_lenghts"] = torch.count_nonzero(
            batch["context_attention_mask"], dim=1
        )
        (
            batch["subword_start_mask"],
            batch["subword_end_mask"],
        ) = self.find_subword_indexes(cattr[self.ENCODING_ATTR_ID["word_ids"]])

        if not testing:
            batch["answers"] = self.find_tokenized_answer_indexes(
                batch["context_offsets"], answers_start, answers_end
            )
        return batch


class TransformerSquadTokenizer(SquadTokenizer):
    """
    Tokenizer and data collator to be used with
    a Transformer-based model, which uses a single
    tokenizer (for the concatenation of question and context)
    """

    def __init__(self, tokenizer, device="cpu"):
        super().__init__(device=device)
        assert isinstance(tokenizer, Tokenizer) or isinstance(tokenizer, BaseTokenizer)
        self.tokenizer = tokenizer

    def select_tokenizer(self, entity=None):
        """
        Ignore the entity parameter and always return
        the same tokenizer
        """
        return self.tokenizer

    def __call__(self, inputs):
        zipped_inputs = tuple(zip(*inputs))
        if len(zipped_inputs) > 3:
            (indexes, questions, contexts, answers_start, answers_end) = zipped_inputs
            testing = False
        else:
            (indexes, questions, contexts) = zipped_inputs
            testing = True

        tokenized = self.tokenize(list(zip(questions, contexts)), special=True)
        attr = list(zip(*[self.ATTRGETTER(e) for e in tokenized]))

        # Create the batch dictionary with encoding info
        batch = {
            "context_ids": torch.tensor(
                attr[self.ENCODING_ATTR_ID["ids"]], device=self.device
            ),
            "context_type_ids": torch.tensor(
                attr[self.ENCODING_ATTR_ID["type_ids"]], device=self.device
            ),
            "attention_mask": torch.tensor(
                attr[self.ENCODING_ATTR_ID["attention_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "special_tokens_mask": torch.tensor(
                attr[self.ENCODING_ATTR_ID["special_tokens_mask"]],
                dtype=torch.bool,
                device=self.device,
            ),
            "offsets": torch.tensor(
                attr[self.ENCODING_ATTR_ID["offsets"]], device=self.device
            ),
            "indexes": torch.tensor(indexes, dtype=torch.long, device=self.device),
        }

        # Add custom info to the batch dict
        batch["context_attention_mask"] = (
            batch["context_type_ids"].bool() & ~batch["special_tokens_mask"]
        )
        batch["context_offsets"] = torch.where(
            batch["context_attention_mask"].unsqueeze(-1).repeat(1, 1, 2),
            batch["offsets"],
            -100,
        )
        (
            batch["subword_start_mask"],
            batch["subword_end_mask"],
        ) = self.find_subword_indexes(attr[self.ENCODING_ATTR_ID["word_ids"]])

        if not testing:
            batch["answers"] = self.find_tokenized_answer_indexes(
                batch["context_offsets"], answers_start, answers_end
            )
        return batch


def get_transformer_tokenizer(
    vocab_path="data/bert-base-uncased-vocab.txt", max_tokens=512, device="cpu"
):
    """
    Return a tokenizer to be used with Transformer-based models
    """
    wp_tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=True)
    wp_tokenizer.enable_padding(direction="right", pad_type_id=1)
    wp_tokenizer.enable_truncation(max_tokens)
    return TransformerSquadTokenizer(wp_tokenizer, device=device)


def get_recurrent_tokenizer(
    vocab, max_context_tokens, unk_token="[UNK]", pad_token="[PAD]", device="cpu"
):
    """
    Return a tokenizer to be used with recurrent-based models
    """
    question_tokenizer = Tokenizer(WordLevel(vocab, unk_token=unk_token))
    question_tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
    question_tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
    question_tokenizer.enable_padding(
        direction="right", pad_id=vocab[pad_token], pad_type_id=1, pad_token=pad_token
    )

    context_tokenizer = Tokenizer(WordLevel(vocab, unk_token=unk_token))
    context_tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
    context_tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
    context_tokenizer.enable_padding(
        direction="right", pad_id=vocab[pad_token], pad_type_id=1, pad_token=pad_token,
    )
    context_tokenizer.enable_truncation(max_context_tokens)

    return RecurrentSquadTokenizer(question_tokenizer, context_tokenizer, device=device)
