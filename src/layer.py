from collections import OrderedDict

import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils
import layer_utils


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
        return layer_utils.masked_softmax(
            x, self.dim, mask=mask, log=self.log, eps=self.eps, device=self.device
        )


class LSTM(nn.Module):
    """
    Generic LSTM module which handles padded sequences
    and dropout (both inside and outside cells)
    """

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


class Highway(nn.Module):
    """
    Generic Highway network
    
    "Highway Networks",
    Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    """

    def __init__(
        self, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid, device="cpu"
    ):
        super().__init__()
        # Hidden and gate layers
        self.linear = nn.Sequential(nn.Linear(size, size), nonlinearity())
        self.gate = nn.Sequential(nn.Linear(size, size), gate_activation())

        # Transfer to device
        self.device = device
        self.to(self.device)

    def forward(self, x):
        h = self.linear(x)
        g = self.gate(x)
        return g * h + (1 - g) * x


class AttentionFlow(nn.Module):
    """
    Attention flow layer to be used in the BiDAF model,
    which uses context to query and query to context
    attention types
    """

    def __init__(self, size, device="cpu"):
        super().__init__()
        self.size = size
        self.similarity = nn.Linear(3 * size, 1, bias=False)
        self.query_aware_context = nn.Linear(4 * size, 4 * size)

        # Transfer to device
        self.device = device
        self.to(self.device)

    def get_similarity_input(self, questions, contexts):
        """
        Both C2Q and Q2C attentions share a similarity matrix, 
        between the contextual embeddings of the context and the query, 
        where each element indicates the similarity between 
        a context word and a query word
        """
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
        """
        Context-to-query (C2Q) attention signifies which query words are
        most relevant to each context word
        """
        return torch.bmm(
            layer_utils.masked_softmax(
                sim, 2, mask=questions_mask.unsqueeze(1), device=self.device
            ),
            questions,
        )

    def query_to_context(self, sim, contexts, contexts_mask):
        """
        Query-to-context (Q2C) attention signifies which context words
        have the closest similarity to one of the query words 
        and are hence critical for answering the query
        """
        sim_max_col = sim.max(dim=2, keepdims=True)
        attention_weights = layer_utils.masked_softmax(
            sim_max_col.values, 1, mask=contexts_mask.unsqueeze(-1), device=self.device
        )
        return torch.bmm(attention_weights.transpose(1, 2), contexts).repeat(
            1, contexts.shape[1], 1
        )

    def forward(self, questions, contexts, questions_mask, contexts_mask):
        # Get the input similarity matrix
        input_sim = self.get_similarity_input(questions, contexts)

        # Similarity matrix (S)
        sim = self.similarity(input_sim).squeeze(-1)

        # Matrix that contains the attended query vectors for the entire context (U tilde)
        u_tilde = self.context_to_query(sim, questions, questions_mask)

        # Matrix that indicates the the most important words in the
        # context with respect to the query (H tilde)
        h_tilde = self.query_to_context(sim, contexts, contexts_mask)

        # Mega-merge
        megamerge = torch.cat(
            [contexts, u_tilde, u_tilde * contexts, h_tilde * contexts], dim=2
        )
        g = self.query_aware_context(megamerge)
        return g


class QAOutput(nn.Module):
    """
    Generic output layer to be used for
    question answering tasks
    """

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

        # Whether to mask the input sequence
        # (to ignore padding) or not
        self.masked = masked

        # Start and end token classifiers
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
        """
        Given start and end tokens probabilities,
        return the most probable answer span (in tokens)
        """
        # Compute the joint probability distribution
        p_joint = torch.einsum("bi,bj->bij", (start_probs, end_probs))
        is_legal_pair = torch.triu(
            torch.ones((start_probs.shape[1], end_probs.shape[1]), device=self.device)
        )
        p_joint *= is_legal_pair

        # Take pair (i, j) that maximizes the joint probability
        max_in_row, _ = torch.max(p_joint, dim=2)
        max_in_col, _ = torch.max(p_joint, dim=1)
        start_indexes = torch.argmax(max_in_row, dim=-1).unsqueeze(-1)
        end_indexes = torch.argmax(max_in_col, dim=-1).unsqueeze(-1)

        return torch.cat([start_indexes, end_indexes], dim=-1)

    def from_token_to_char(self, context_offsets, output_indexes):
        """
        Convert the given token answer span to a word answer span
        """
        word_start_indexes = torch.gather(
            context_offsets[:, :, 0], 1, output_indexes[:, 0].unsqueeze(-1)
        )
        word_end_indexes = torch.gather(
            context_offsets[:, :, 1], 1, output_indexes[:, 1].unsqueeze(-1)
        )
        return torch.cat([word_start_indexes, word_end_indexes], dim=1)

    def forward(self, start_input, end_input, **inputs):
        # Compute start token probabilities
        start_indexes = self.start_classifier(start_input).squeeze(-1)
        masked_start = inputs["context_attention_mask"] & inputs["subword_start_mask"]
        start_probs = self.softmax(
            self.dropout(start_indexes), masked_start if self.masked else None
        )

        # Compute end token probabilities
        end_indexes = self.end_classifier(end_input).squeeze(-1)
        masked_end = inputs["subword_end_mask"] & inputs["context_attention_mask"]
        end_probs = self.softmax(
            self.dropout(end_indexes), masked_end if self.masked else None
        )

        # Get token and word answer spans
        outputs = self.get_qa_outputs(torch.exp(start_probs), torch.exp(end_probs))
        word_outputs = self.from_token_to_char(inputs["context_offsets"], outputs)

        # If labels are present, compute loss
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

        # If labels are not present, just return outputs
        return OrderedDict(
            {"word_outputs": word_outputs, "indexes": inputs["indexes"],}
        )
