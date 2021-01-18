from collections import OrderedDict

import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import utils


class MaskedSoftmax(nn.Module):
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
        if mask is None:
            mask = torch.ones_like(x, device=self.device)
        exp = torch.exp(x) * torch.where(
            mask,
            mask.float(),
            torch.tensor(self.eps, dtype=torch.float32, device=self.device),
        )
        softmax = exp / exp.sum(dim=self.dim).unsqueeze(-1)
        return softmax if not self.log else torch.log(softmax)


class QAOutput(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout_rate=0.2,
        classifier_bias=True,
        device="cpu",
    ):
        super().__init__()

        self.start_classifier = nn.Linear(input_size, output_size, bias=classifier_bias)
        self.end_classifier = nn.Linear(input_size, output_size, bias=classifier_bias)

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Loss criterion
        self.softmax = MaskedSoftmax(dim=1, log=True, device=device)
        self.criterion = nn.NLLLoss(reduction="mean", ignore_index=-1)

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def get_qa_outputs(self, start_probs, end_probs):
        end_indexes = end_probs.argmax(dim=1, keepdims=True)
        indexes = torch.stack(
            start_probs.shape[0] * [torch.arange(start_probs.shape[1])]
        )
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
        end_indexes = self.end_classifier(end_input).squeeze(-1)
        masked_end = inputs["subword_end_mask"] & inputs["context_attention_mask"]
        end_probs = self.softmax(self.dropout(end_indexes), masked_end)

        ########### inputs["context_attention_mask"] to start_probs or masked_start?
        end_best_indexes = end_probs.argmax(dim=1, keepdims=True)
        indexes = torch.stack(end_probs.shape[0] * [torch.arange(end_probs.shape[1])])
        masked_start = (
            (indexes <= end_best_indexes)
            & inputs["context_attention_mask"]
            & inputs["subword_start_mask"]
        )
        ############

        start_indexes = self.start_classifier(start_input).squeeze(-1)
        start_probs = self.softmax(
            self.dropout(start_indexes),
            masked_start,  # inputs["context_attention_mask"] ???
        )

        outputs = self.get_qa_outputs(start_probs, end_probs)
        answers = utils.get_nearest_answers(
            inputs["answers"], outputs, device=self.device
        )

        start_loss = self.criterion(start_probs, answers[:, 0])
        end_loss = self.criterion(end_probs, answers[:, 1])
        loss = start_loss + end_loss

        word_outputs = self.from_token_to_char(inputs["context_offsets"], outputs)
        return OrderedDict(
            {
                "loss": loss,
                "token_outputs": outputs,
                "word_outputs": word_outputs,
                "indexes": inputs["indexes"],
            }
        )


class QABaselineModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        max_context_tokens,
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

        # Strategy to perform sentence embedding
        self.recurrent_module = nn.LSTM(
            self.embedding_dimension,
            self.embedding_dimension,
            batch_first=True,
            num_layers=num_recurrent_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )

        # Output layer
        self.output_layer = QAOutput(
            self.embedding_dimension * 2,
            max_context_tokens,
            dropout_rate=dropout_rate,
            classifier_bias=True,
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

    def _sentence_embedding(
        self, questions, contexts, question_lenghts, context_lenghts
    ):
        """
        Use an RNN to encode tokens and extract information from the RNN states:
        average all the output layers
        """
        packed_questions = pack_padded_sequence(
            questions, question_lenghts, batch_first=True, enforce_sorted=False
        )
        packed_contexts = pack_padded_sequence(
            contexts, context_lenghts, batch_first=True, enforce_sorted=False,
        )
        output_questions, (hidden, cell) = self.recurrent_module(packed_questions)
        output_contexts, (hidden, cell) = self.recurrent_module(packed_contexts)
        padded_questions, padded_questions_lenghts = pad_packed_sequence(
            output_questions, batch_first=True
        )
        padded_contexts, padded_contexts_lenghts = pad_packed_sequence(
            output_contexts, batch_first=True
        )
        return (
            padded_questions.sum(dim=1) / padded_questions_lenghts.view(-1, 1),
            padded_contexts.sum(dim=1) / padded_contexts_lenghts.view(-1, 1),
        )

    def _merge_embeddings(self, questions, contexts):
        """
        Merge the given embeddings concatenating them
        """
        return torch.cat([questions, contexts], dim=1)

    def forward(self, **inputs):
        """
        Perform a forward pass and return predictions over
        a mini-batch of sequences of the same lenght
        """
        embedded_questions = self.embedding(inputs["question_ids"])
        embedded_contexts = self.embedding(inputs["context_ids"])
        sentence_questions, sentence_contexts = self._sentence_embedding(
            embedded_questions,
            embedded_contexts,
            inputs["question_lenghts"],
            inputs["context_lenghts"],
        )
        merged_inputs = self._merge_embeddings(sentence_questions, sentence_contexts)

        return self.output_layer(merged_inputs, merged_inputs, **inputs)


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

    def context_to_query(self, sim, questions):
        return torch.bmm(F.softmax(sim, dim=2), questions)

    def query_to_context(self, sim, contexts):
        sim_max_col = sim.max(dim=2, keepdims=True)
        attention_weights = F.softmax(sim_max_col.values, dim=1)
        return torch.bmm(attention_weights.transpose(1, 2), contexts).repeat(
            1, contexts.shape[1], 1
        )

    def forward(self, questions, contexts):
        input_sim = self.get_similarity_input(questions, contexts)

        # S
        sim = self.similarity(input_sim).squeeze(-1)

        # U tilde
        u_tilde = self.context_to_query(sim, questions)

        # H tilde
        h_tilde = self.query_to_context(sim, contexts)

        # Megamerge
        megamerge = torch.cat(
            [contexts, u_tilde, u_tilde * contexts, h_tilde * contexts], dim=2
        )
        g = self.query_aware_context(megamerge)

        return g


class QABiDAFModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        highway_depth=2,
        dropout_rate=0.2,
        contextual_recurrent_layers=2,
        contextual_bidirectional=False,
        device="cpu",
    ):
        """

        """
        super().__init__()

        # Embedding module
        self.word_embedding = embedding_module
        self.word_embedding_dimension = embedding_module.weight.shape[-1]

        # Highway network
        self.highway_depth = highway_depth
        self.highway = get_highway(
            self.highway_depth, self.word_embedding_dimension, device=device
        )

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Contextual embeddings
        self.contextual_embedding = nn.LSTM(
            self.word_embedding_dimension,
            self.word_embedding_dimension,
            batch_first=True,
            num_layers=contextual_recurrent_layers,
            bidirectional=contextual_bidirectional,
            dropout=self.dropout_rate if contextual_recurrent_layers > 1 else 0.0,
        )

        # Attention flow
        self.attention = AttentionFlow(self.word_embedding_dimension, device=device)

        # Modeling layer
        self.modeling_layer = nn.LSTM(
            4 * self.word_embedding_dimension,
            self.word_embedding_dimension,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=self.dropout_rate,
        )

        # Output layer
        self.out_lstm = nn.LSTM(
            2 * self.word_embedding_dimension,
            self.word_embedding_dimension,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = QAOutput(
            6 * self.word_embedding_dimension,
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
        embedded_questions = self.word_embedding(inputs["question_ids"].to(self.device))
        embedded_contexts = self.word_embedding(inputs["context_ids"].to(self.device))

        highway_questions = self.highway(embedded_questions)
        highway_contexts = self.highway(embedded_contexts)

        contextual_questions = self.dropout(
            self.contextual_embedding(highway_questions)[0]
        )
        contextual_contexts = self.dropout(
            self.contextual_embedding(highway_contexts)[0]
        )

        query_aware_contexts = self.attention(contextual_questions, contextual_contexts)

        modeling = self.dropout(self.modeling_layer(query_aware_contexts)[0])
        m2 = self.dropout(self.out_lstm(modeling)[0])

        return self.output_layer(
            torch.cat([query_aware_contexts, modeling], dim=-1),
            torch.cat([query_aware_contexts, m2], dim=-1),
            **inputs,
        )


class QABertModel(nn.Module):
    def __init__(self, dropout_rate=0.2, device="cpu"):
        super().__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.output_layer = QAOutput(
            768, 1, dropout_rate=dropout_rate, classifier_bias=True, device=device
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
        outputs = self.output_layer(bert_outputs, bert_outputs, **inputs)
        return outputs
