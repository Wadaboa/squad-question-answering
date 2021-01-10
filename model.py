import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class QABaselineModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        max_context_tokens,
        num_recurrent_layers=2,
        bidirectional=False,
        dropout=0.2,
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
            dropout=dropout,
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
        start_indexes = start_probs.argmax(dim=1, keepdims=True)
        end_indexes = end_probs.argmax(dim=1, keepdims=True)

        word_start_indexes = torch.gather(
            inputs["context_offsets"][:, :, 0], 1, start_indexes
        )
        word_end_indexes = torch.gather(
            inputs["context_offsets"][:, :, 1], 1, end_indexes
        )
        outputs = torch.cat([word_start_indexes, word_end_indexes], dim=1)

        start_loss = self.criterion(start_probs, inputs["answer_start"].to(self.device))
        end_loss = self.criterion(end_probs, inputs["answer_end"].to(self.device))

        return {"loss": start_loss + end_loss, "outputs": outputs}


class Highway(nn.Module):
    def __init__(self, size, nonlinearity=nn.ReLU, gate_activation=F.sigmoid):
        super(Highway, self).__init__()
        self.size = size
        self.nonlinearity = nonlinearity
        self.gate_activation = gate_activation

        self.linear = nn.Sequential(nn.Linear(size, size), self.nonlinearity)
        self.gate = nn.Sequential(nn.Linear(size, size), self.gate_activation)

    def forward(self, x):
        h = self.linear(x)
        g = self.gate(x)
        return g * h + (1 - g) * x


def get_highway(num_layers, size, nonlinearity=nn.ReLU, gate_activation=F.sigmoid):
    highway = []
    for _ in range(num_layers):
        highway.append(
            Highway(size, nonlinearity=nonlinearity, gate_activation=gate_activation)
        )
    return nn.Sequential(*highway)


class AttentionFlow(nn.Module):
    def __init__(self, size):
        self.size = size
        self.similarity = nn.Linear(3 * size, 1, bias=False)
        self.query_aware_context = nn.Linear(4 * size, 4 * size)
    
    def get_similarity_input(self, questions, contexts)
        questions_shape = questions.shape
        contexts_shape = contexts.shape
        assert questions_shape[0] == contexts_shape[0], "Batch size must be equal"
        assert questions_shape[2] == contexts_shape[2], "Embedding size must be equal"
        input_sim_shape = view_shape[:-1] + (questions_shape[2] * 3,)
        view_shape = (questions_shape[0], contexts_shape[1], questions_shape[1], questions_shape[2])
        input_sim = torch.cat(
            [
                torch.repeat_interleave(contexts, questions_shape[1], dim=1).view(*view_shape),
                questions.repeat(1, contexts_shape[1], 1).view(*view_shape),
                torch.einsum("bcd, bqd->bcqd", contexts, questions),
            ],
            dim=3,
        )
        assert input_sim.shape == input_sim_shape, f"Wrong similarity matrix input shape {input_sim.shape}"
        return input_sim
    
    def context_to_query(self, sim, questions):
        return torch.bmm(F.softmax(sim, dim=1), questions).transpose(1,2)
    
    def query_to_context(self, sim, contexts):
        sim_max_col = sim.max(dim=1, keepdims=True)
        attention_weights = F.softmax(sim_max_col.values, dim=0)
        return torch.bmm(attention_weights.transpose(1,2), contexts).repeat(1, contexts.shape[1], 1)
    
    def forward(self, questions, contexts):
        input_sim = self.get_similarity_input(questions, contexts)
        
        # S
        sim = self.similarity(input_sim).squeeze(-1)
        
        # U tilde
        ctq = self.context_to_query(sim, questions)
        
        # H tilde
        qtc = self.query_to_context(sim, contexts)
        


class CustomBiDAFModel(nn.Module):
    def __init__(
        self,
        embedding_module,
        max_context_tokens,
        highway_depth=2,
        num_recurrent_layers=2,
        bidirectional=False,
        dropout=0.2,
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

        # Contextual embeddings
        self.contextual_embedding = nn.LSTM(
            self.word_embedding_dimension,
            self.word_embedding_dimension,
            batch_first=True,
            num_layers=num_recurrent_layers,
            bidirectional=bidirectional,
            dropout=dropout,
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

        # Transfer model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def count_parameters(self):
        """
        Return the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, **inputs):
        embedded_questions = self.embedding(inputs["question_ids"].to(self.device))
        embedded_contexts = self.embedding(inputs["context_ids"].to(self.device))

        highway_questions = self.highway(embedded_questions)
        highway_contexts = self.highway(embedded_contexts)

        contextual_questions = self.contextual_embedding(highway_questions)
        contextual_contexts = self.contextual_embedding(highway_contexts)
