import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class QAOutput(nn.Module):
    def __init__(
        self,
        size,
        dropout_rate=0.2,
        classifier_bias=True,
        alpha_coeff=1.0,
        beta_coeff=0.0,
    ):
        super().__init__()

        self.start_classifier = nn.Linear(size, 1, bias=classifier_bias)
        self.end_classifier = nn.Linear(size, 1, bias=classifier_bias)

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Loss criterion
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=-1)
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff

        # Transfer model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_qa_outputs(self, start_probs, end_probs, context_offsets, mask):
        masked_end = torch.where(mask, end_probs, torch.tensor(-1, dtype=torch.float32))
        end_indexes = masked_end.argmax(dim=1, keepdims=True)

        indexes = torch.stack(
            start_probs.shape[0] * [torch.arange(start_probs.shape[1])]
        )
        masked_start = torch.where(indexes <= end_indexes, start_probs, torch.tensor(-1, dtype=torch.float32))
        start_indexes = masked_start.argmax(dim=1, keepdims=True)

        return torch.cat([start_indexes, end_indexes], dim=1)

    def from_token_to_char(self, context_offsets, output_indexes):
        word_start_indexes = torch.gather(
            context_offsets[:, :, 0], 1, output_indexes[:, 0]
        )
        word_end_indexes = torch.gather(
            context_offsets[:, :, 1], 1, output_indexes[:, 1]
        )
        return torch.cat([word_start_indexes, word_end_indexes], dim=1)

    def get_nearest_answers(self, outputs, answers):
        distances = torch.where(
            answers != -1,
            torch.abs(outputs.unsqueeze(-1).repeat(1, 1, answers.shape[-1]) - answers),
            -1,
        )
        summed_distances = torch.sum(distances, dim=1, keepdims=True)
        min_indexes = torch.min(summed_distances, dim=0, keepdims=True)
        return answers[min_indexes]

    def precedence_constraint(self, start_probs, end_probs):
        num_tokens = start_probs.shape[1]
        best_start = start_probs.argmax(dim=1, keepdims=True) / num_tokens
        best_end = start_probs.argmax(dim=1, keepdims=True) / num_tokens
        return torch.abs(torch.clip(best_end - best_start, 0, num_tokens))

    def forward(self, start_input, end_input, **inputs):
        start_indexes = self.start_classifier(start_input).squeeze(-1)
        start_probs = self.softmax(self.dropout(start_indexes))

        end_indexes = self.end_classifier(end_input).squeeze(-1)
        end_probs = self.softmax(self.dropout(end_indexes))

        outputs = self.get_qa_outputs(
            start_probs,
            end_probs,
            inputs["context_offsets"],
            inputs["context_attention_mask"],
        )
        answers = self.get_nearest_answers(outputs, inputs["answers"])

        start_loss = self.criterion(start_probs, answers[:, 0])
        end_loss = self.criterion(end_probs, answers[:, 1])
        cst = self.precedence_constraint(start_probs, end_probs)
        loss = self.alpha_coeff * (start_loss + end_loss) + self.beta_coeff * cst

        word_outputs = self.from_token_to_char(
            inputs["context_offsets"], outputs[:, 0], outputs[:, 1]
        )

        return {"loss": loss, "outputs": outputs}


class QABaselineModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        max_context_tokens,
        num_recurrent_layers=2,
        bidirectional=False,
        dropout_rate=0.2,
        alpha_coeff=1.0,
        beta_coeff=0.0,
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
        
        self.output_layer = QAOutput(
            self.embedding_dimension * 2,
            dropout_rate=dropout_rate,
            classifier_bias=True,
            alpha_coeff=alpha_coeff,
            beta_coeff=beta_coeff,
        )
        
        # Classification layer
        self.start_classifier = nn.Linear(
            self.embedding_dimension * 2, max_context_tokens
        )
        self.end_classifier = nn.Linear(
            self.embedding_dimension * 2, max_context_tokens
        )

        # Loss criterion
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=-1)
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff

        # Transfer model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        embedded_questions = self.embedding(inputs["question_ids"].to(self.device))
        embedded_contexts = self.embedding(inputs["context_ids"].to(self.device))
        sentence_questions, sentence_contexts = self._sentence_embedding(
            embedded_questions,
            embedded_contexts,
            inputs["question_lenghts"].to(self.device),
            inputs["context_lenghts"].to(self.device),
        )
        merged_inputs = self._merge_embeddings(sentence_questions, sentence_contexts)
        
        
        
        
        start_probs = self.softmax(self.start_classifier(merged_inputs))
        end_probs = self.softmax(self.end_classifier(merged_inputs))

        start_loss = self.criterion(start_probs, inputs["answer_start"].to(self.device))
        end_loss = self.criterion(end_probs, inputs["answer_end"].to(self.device))
        cst = precedence_constraint(start_probs, end_probs)
        loss = self.alpha_coeff * (start_loss + end_loss) + self.beta_coeff * cst

        outputs = get_qa_outputs(start_probs, end_probs, inputs["context_offsets"])

        return {"loss": loss, "outputs": outputs}


class Highway(nn.Module):
    def __init__(self, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(size, size), nonlinearity())
        self.gate = nn.Sequential(nn.Linear(size, size), gate_activation())

    def forward(self, x):
        h = self.linear(x)
        g = self.gate(x)
        return g * h + (1 - g) * x


def get_highway(num_layers, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid):
    highway = []
    for _ in range(num_layers):
        highway.append(
            Highway(size, nonlinearity=nonlinearity, gate_activation=gate_activation)
        )
    return nn.Sequential(*highway)


class AttentionFlow(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.similarity = nn.Linear(3 * size, 1, bias=False)
        self.query_aware_context = nn.Linear(4 * size, 4 * size)

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


class BiDAFModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        highway_depth=2,
        dropout_rate=0.2,
        contextual_recurrent_layers=2,
        contextual_bidirectional=False,
        alpha_coeff=1.0,
        beta_coeff=0.0,
    ):
        """
        
        """
        super().__init__()

        # Embedding module
        self.word_embedding = embedding_module
        self.word_embedding_dimension = embedding_module.weight.shape[-1]

        # Highway network
        self.highway_depth = highway_depth
        self.highway = get_highway(self.highway_depth, self.word_embedding_dimension)

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
        self.attention = AttentionFlow(self.word_embedding_dimension)

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
            dropout_rate=self.dropout_rate,
            classifier_bias=False,
            alpha_coeff=alpha_coeff,
            beta_coeff=beta_coeff,
        )

        # Transfer model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
