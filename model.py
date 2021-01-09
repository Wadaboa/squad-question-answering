import torch
import torch.nn as nn
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
