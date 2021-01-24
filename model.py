from collections import OrderedDict

import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import utils


def get_embedding_module(embedding_model, pad_id):
    """
    Given a Gensim embedding model, load the weight matrix
    into a PyTorch Embedding module and set it as non-trainable
    """
    embedding_layer = nn.Embedding(
        embedding_model.vectors.shape[0],
        embedding_model.vectors.shape[1],
        padding_idx=pad_id,
    )
    embedding_layer.weight = nn.Parameter(torch.from_numpy(embedding_model.vectors))
    embedding_layer.weight.requires_grad = False
    return embedding_layer


def masked_softmax(x, dim, mask=None, log=False, eps=1e-4, device="cpu"):
    """
    Functional version of a softmax with masked inputs
    """
    if mask is None:
        mask = torch.ones_like(x, device=device)
    exp = torch.exp(x) * torch.where(
        mask, mask.float(), torch.tensor(eps, dtype=torch.float32, device=device),
    )
    softmax = exp / exp.sum(dim=dim).unsqueeze(-1)
    return softmax if not log else torch.log(softmax)


class MaskedSoftmax(nn.Module):
    """
    Modular version of a softmax with masked inputs
    """

    def __init__(self, dim, log=False, eps=1e-4, device="cpu"):
        super().__init__()
        self.dim = dim
        self.log = log
        self.eps = eps
        self.device = device

    def forward(self, x, mask=None):
        """
        Performs masked softmax
        """
        return masked_softmax(
            x, self.dim, mask=mask, log=self.log, eps=self.eps, device=self.device
        )


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lenghts):
        packed_inputs = pack_padded_sequence(
            x, lenghts.cpu(), batch_first=self.lstm.batch_first, enforce_sorted=False
        )
        output, (hidden, cell) = self.lstm(packed_inputs)
        padded_outputs, padded_output_lenghts = pad_packed_sequence(
            output, batch_first=self.lstm.batch_first
        )
        padded_outputs = self.dropout(padded_outputs)
        return padded_outputs, padded_output_lenghts


class QAOutput(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_rate=0.2,
        classifier_bias=True,
        masked=True,
        device="cpu",
    ):
        super().__init__()

        self.masked = masked

        self.start_classifier = nn.Linear(input_size, output_size, bias=classifier_bias)
        self.end_classifier = nn.Linear(input_size, output_size, bias=classifier_bias)

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Loss criterion
        self.softmax = MaskedSoftmax(dim=1, log=True, device=device)
        self.criterion = nn.NLLLoss(reduction="mean", ignore_index=-100)

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def get_qa_outputs(self, start_probs, end_probs):
        end_indexes = end_probs.argmax(dim=1, keepdims=True)
        indexes = torch.stack(
            start_probs.shape[0] * [torch.arange(start_probs.shape[1])]
        ).to(self.device)
        masked_start = torch.where(
            indexes <= end_indexes,
            start_probs,
            torch.tensor(-np.inf, dtype=torch.float32, device=self.device),
        )
        start_indexes = masked_start.argmax(dim=1, keepdims=True)

        return torch.cat([start_indexes, end_indexes], dim=1)

    def from_token_to_char(self, context_offsets, output_indexes):
        word_start_indexes = torch.gather(
            context_offsets[:, :, 0], 1, output_indexes[:, 0].unsqueeze(-1)
        )
        word_end_indexes = torch.gather(
            context_offsets[:, :, 1], 1, output_indexes[:, 1].unsqueeze(-1)
        )
        return torch.cat([word_start_indexes, word_end_indexes], dim=1)

    def forward(self, start_input, end_input, **inputs):
        start_indexes = self.start_classifier(start_input).squeeze(-1)
        masked_start = inputs["context_attention_mask"] & inputs["subword_start_mask"]
        start_probs = self.softmax(
            self.dropout(start_indexes), masked_start if self.masked else None
        )

        end_indexes = self.end_classifier(end_input).squeeze(-1)
        masked_end = inputs["subword_end_mask"] & inputs["context_attention_mask"]
        end_probs = self.softmax(
            self.dropout(end_indexes), masked_end if self.masked else None
        )

        outputs = self.get_qa_outputs(start_probs, end_probs)
        word_outputs = self.from_token_to_char(inputs["context_offsets"], outputs)

        if "answers" in inputs:
            answers = utils.get_nearest_answers(
                inputs["answers"], outputs, device=self.device
            )
            start_loss = self.criterion(start_probs, answers[:, 0])
            end_loss = self.criterion(end_probs, answers[:, 1])
            loss = (start_loss + end_loss) / 2
            return OrderedDict(
                {
                    "loss": loss,
                    "word_outputs": word_outputs,
                    "indexes": inputs["indexes"],
                }
            )

        return OrderedDict(
            {"word_outputs": word_outputs, "indexes": inputs["indexes"],}
        )


class QABaselineModel(nn.Module):
    IGNORE_LAYERS = ["embedding.weight"]

    def __init__(
        self,
        embedding_module,
        max_context_tokens,
        hidden_size=100,
        num_recurrent_layers=2,
        bidirectional=False,
        dropout_rate=0.2,
        device="cpu",
    ):
        """
        Build a generic question answering model, with recurrent modules
        """
        super().__init__()

        # Embedding module
        self.embedding = embedding_module
        self.embedding_dimension = embedding_module.weight.shape[-1]

        self.projection = nn.Linear(self.embedding_dimension, hidden_size)

        # Strategy to perform sentence embedding
        self.recurrent_module = LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_recurrent_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )

        # Output layer
        out_dim = hidden_size if not bidirectional else 2 * hidden_size
        self.out_lstm = LSTM(
            out_dim, hidden_size, batch_first=True, bidirectional=bidirectional
        )
        self.output_layer = QAOutput(
            out_dim, 1, dropout_rate=dropout_rate, classifier_bias=True, device=device,
        )

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def count_parameters(self):
        """
        Return the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, **inputs):
        """
        Perform a forward pass and return predictions over
        a mini-batch of sequences of the same lenght
        """
        embedded_questions = self.embedding(inputs["question_ids"])
        embedded_contexts = self.embedding(inputs["context_ids"])

        hidden_questions = self.projection(embedded_questions)
        hidden_contexts = self.projection(embedded_contexts)

        padded_questions, padded_questions_lenghts = self.recurrent_module(
            hidden_questions, inputs["question_lenghts"]
        )
        padded_contexts, _ = self.recurrent_module(
            hidden_contexts, inputs["context_lenghts"]
        )

        average_questions = padded_questions.sum(dim=1) / inputs[
            "question_lenghts"
        ].view(-1, 1)
        start_input = padded_contexts * average_questions.unsqueeze(1).repeat(
            1, padded_contexts.shape[1], 1
        )
        end_input, _ = self.out_lstm(start_input, inputs["context_lenghts"])

        return self.output_layer(start_input, end_input, **inputs)

    def state_dict(self):
        st_dict = super().state_dict()
        return {k: st_dict[k] for k in st_dict.keys() if k not in self.IGNORE_LAYERS}

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=strict)


class Highway(nn.Module):
    def __init__(
        self, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid, device="cpu"
    ):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(size, size), nonlinearity())
        self.gate = nn.Sequential(nn.Linear(size, size), gate_activation())

        self.device = device
        self.to(self.device)

    def forward(self, x):
        h = self.linear(x)
        g = self.gate(x)
        return g * h + (1 - g) * x


def get_highway(
    num_layers, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid, device="cpu"
):
    highway = []
    for _ in range(num_layers):
        highway.append(
            Highway(
                size,
                nonlinearity=nonlinearity,
                gate_activation=gate_activation,
                device=device,
            )
        )
    return nn.Sequential(*highway)


class AttentionFlow(nn.Module):
    def __init__(self, size, device="cpu"):
        super().__init__()
        self.size = size
        self.similarity = nn.Linear(3 * size, 1, bias=False)
        self.query_aware_context = nn.Linear(4 * size, 4 * size)

        self.device = device
        self.to(self.device)

    def get_similarity_input(self, questions, contexts):
        questions_shape = questions.shape
        contexts_shape = contexts.shape
        assert questions_shape[0] == contexts_shape[0], "Batch size must be equal"
        assert questions_shape[2] == contexts_shape[2], "Embedding size must be equal"
        view_shape = (
            questions_shape[0],
            contexts_shape[1],
            questions_shape[1],
            questions_shape[2],
        )
        input_sim_shape = view_shape[:-1] + (questions_shape[2] * 3,)
        input_sim = torch.cat(
            [
                torch.repeat_interleave(contexts, questions_shape[1], dim=1).view(
                    *view_shape
                ),
                questions.repeat(1, contexts_shape[1], 1).view(*view_shape),
                torch.einsum("bcd, bqd->bcqd", contexts, questions),
            ],
            dim=3,
        )
        assert (
            input_sim.shape == input_sim_shape
        ), f"Wrong similarity matrix input shape {input_sim.shape}"
        return input_sim

    def context_to_query(self, sim, questions, questions_mask):
        return torch.bmm(
            masked_softmax(
                sim, 2, mask=questions_mask.unsqueeze(1), device=self.device
            ),
            questions,
        )

    def query_to_context(self, sim, contexts, contexts_mask):
        sim_max_col = sim.max(dim=2, keepdims=True)
        attention_weights = masked_softmax(
            sim_max_col.values, 1, mask=contexts_mask.unsqueeze(-1), device=self.device
        )
        return torch.bmm(attention_weights.transpose(1, 2), contexts).repeat(
            1, contexts.shape[1], 1
        )

    def forward(self, questions, contexts, questions_mask, contexts_mask):
        input_sim = self.get_similarity_input(questions, contexts)

        # S
        sim = self.similarity(input_sim).squeeze(-1)

        # U tilde
        u_tilde = self.context_to_query(sim, questions, questions_mask)

        # H tilde
        h_tilde = self.query_to_context(sim, contexts, contexts_mask)

        # Megamerge
        megamerge = torch.cat(
            [contexts, u_tilde, u_tilde * contexts, h_tilde * contexts], dim=2
        )
        g = self.query_aware_context(megamerge)

        return g


class QABiDAFModel(nn.Module):
    IGNORE_LAYERS = ["word_embedding.weight"]

    def __init__(
        self,
        embedding_module,
        hidden_size=100,
        highway_depth=2,
        dropout_rate=0.2,
        device="cpu",
    ):
        """"""
        super().__init__()

        # Embedding module
        self.word_embedding = embedding_module
        self.word_embedding_dimension = embedding_module.weight.shape[-1]
        self.projection = nn.Linear(
            self.word_embedding_dimension, hidden_size, bias=False
        )

        # Highway network
        self.highway_depth = highway_depth
        self.highway = get_highway(self.highway_depth, hidden_size, device=device)

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Contextual embeddings
        self.contextual_embedding = LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )

        # Attention flow
        self.attention = AttentionFlow(2 * hidden_size, device=device)

        # Modeling layer
        self.modeling_layer = LSTM(
            8 * hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=self.dropout_rate,
        )

        # Output layer
        self.out_lstm = LSTM(
            2 * hidden_size, hidden_size, batch_first=True, bidirectional=True,
        )
        self.output_layer = QAOutput(
            10 * hidden_size,
            1,
            dropout_rate=self.dropout_rate,
            classifier_bias=False,
            device=device,
        )

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def count_parameters(self):
        """
        Return the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, **inputs):
        questions_mask = inputs["question_attention_mask"]
        contexts_mask = inputs["context_attention_mask"]
        questions_lenght = inputs["question_lenghts"]
        contexts_lenght = inputs["context_lenghts"]

        embedded_questions = self.dropout(self.word_embedding(inputs["question_ids"]))
        embedded_contexts = self.dropout(self.word_embedding(inputs["context_ids"]))

        hidden_questions = self.projection(embedded_questions)
        hidden_contexts = self.projection(embedded_contexts)

        highway_questions = self.highway(hidden_questions)
        highway_contexts = self.highway(hidden_contexts)

        contextual_questions, _ = self.contextual_embedding(
            highway_questions, questions_lenght
        )
        contextual_contexts, _ = self.contextual_embedding(
            highway_contexts, contexts_lenght
        )

        query_aware_contexts = self.attention(
            contextual_questions, contextual_contexts, questions_mask, contexts_mask
        )

        modeling, _ = self.modeling_layer(query_aware_contexts, contexts_lenght)
        m2, _ = self.out_lstm(modeling, contexts_lenght)

        return self.output_layer(
            torch.cat([query_aware_contexts, modeling], dim=-1),
            torch.cat([query_aware_contexts, m2], dim=-1),
            **inputs,
        )

    def state_dict(self):
        st_dict = super().state_dict()
        return {k: st_dict[k] for k in st_dict.keys() if k not in self.IGNORE_LAYERS}

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=strict)


class QABertModel(nn.Module):

    IGNORE_LAYERS = "bert_model"
    BERT_OUTPUT_SIZE = 768

    def __init__(self, dropout_rate=0.2, device="cpu"):
        super().__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out_lstm = LSTM(
            self.BERT_OUTPUT_SIZE,
            self.BERT_OUTPUT_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = QAOutput(
            self.BERT_OUTPUT_SIZE,
            1,
            dropout_rate=dropout_rate,
            classifier_bias=False,
            device=device,
        )

        self.device = device
        self.to(self.device)

    def forward(self, **inputs):
        bert_inputs = {
            "input_ids": inputs["context_ids"],
            "token_type_ids": inputs["context_type_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        bert_outputs = self.bert_model(**bert_inputs)[0]
        end_input, _ = self.out_lstm(bert_outputs)
        outputs = self.output_layer(bert_outputs, end_input, **inputs)
        return outputs

    def count_parameters(self):
        """
        Return the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_dict(self):
        st_dict = super().state_dict()
        return {
            k: st_dict[k]
            for k in st_dict.keys()
            if not k.startswith(self.IGNORE_LAYERS)
        }

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=strict)
