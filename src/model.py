import collections

import transformers
import torch
import torch.nn as nn

import utils
import layer
import layer_utils


class QAModel(nn.Module):
    """
    Abstract question answering module
    """

    IGNORE_LAYERS = []

    def __init__(self):
        super().__init__()

    def count_parameters(self):
        """
        Return the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_dict(self):
        """
        Override default state dict to ignore
        saving pre-defined layers
        """
        st_dict = super().state_dict()
        keys = set(st_dict.keys())
        for l in self.IGNORE_LAYERS:
            for k in st_dict.keys():
                if k.startswith(l):
                    keys.remove(k)
        return collections.OrderedDict({k: v for k, v in st_dict.items() if k in keys})

    def load_state_dict(self, state_dict, strict=False):
        """
        Override default state dict loading
        to allow the non-strict way 
        """
        return super().load_state_dict(state_dict, strict=strict)


class QABaselineModel(QAModel):
    """
    Recurrent encoder with a naive version of attention
    """

    IGNORE_LAYERS = ["embedding.weight"]

    def __init__(
        self,
        embedding_module,
        hidden_size=100,
        num_recurrent_layers=2,
        bidirectional=False,
        dropout_rate=0.2,
        device="cpu",
    ):
        super().__init__()

        # Embedding module
        self.embedding = embedding_module
        self.embedding_dimension = embedding_module.weight.shape[-1]

        # Projection layer (from word embedding dimension to hidden size)
        self.projection = nn.Linear(self.embedding_dimension, hidden_size)

        # Recurrent encoder
        self.recurrent_module = layer.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_recurrent_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )

        # Output layer
        out_dim = hidden_size if not bidirectional else 2 * hidden_size
        self.out_lstm = layer.LSTM(
            out_dim, hidden_size, batch_first=True, bidirectional=bidirectional
        )
        self.output_layer = layer.QAOutput(
            out_dim, 1, dropout_rate=dropout_rate, classifier_bias=True, device=device,
        )

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def forward(self, **inputs):
        # Embed questions and contexts separately
        embedded_questions = self.embedding(inputs["question_ids"])
        embedded_contexts = self.embedding(inputs["context_ids"])

        # Project questions and contexts into a lower-dimensional space
        hidden_questions = self.projection(embedded_questions)
        hidden_contexts = self.projection(embedded_contexts)

        # Call the recurrent encoder for questions and contexts separately
        padded_questions, padded_questions_lenghts = self.recurrent_module(
            hidden_questions, inputs["question_lenghts"]
        )
        padded_contexts, _ = self.recurrent_module(
            hidden_contexts, inputs["context_lenghts"]
        )

        # Average all the hidden states of a single question so as
        # to obtain one vector
        average_questions = padded_questions.sum(dim=1) / inputs[
            "question_lenghts"
        ].view(-1, 1)

        # Perform element-wise multiplication between the aggregated question
        # vector and each vector in the context
        start_input = padded_contexts * average_questions.unsqueeze(1).repeat(
            1, padded_contexts.shape[1], 1
        )
        end_input, _ = self.out_lstm(start_input, inputs["context_lenghts"])
        return self.output_layer(start_input, end_input, **inputs)


class QABiDAFModel(QAModel):
    """
    Custom implementation of a BiDAF model:
    
    "Bidirectional Attention Flow for Machine Comprehension",
    Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    """

    IGNORE_LAYERS = ["word_embedding.weight"]

    def __init__(
        self,
        embedding_module,
        hidden_size=100,
        highway_depth=2,
        dropout_rate=0.2,
        device="cpu",
    ):
        super().__init__()

        # Embedding module
        self.word_embedding = embedding_module
        self.word_embedding_dimension = embedding_module.weight.shape[-1]

        # Projection layer (from word embedding dimension to hidden size)
        self.projection = nn.Linear(
            self.word_embedding_dimension, hidden_size, bias=False
        )

        # Highway network
        self.highway_depth = highway_depth
        self.highway = layer_utils.get_highway(
            self.highway_depth, hidden_size, device=device
        )

        # Dropout module
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        # Contextual embeddings
        self.contextual_embedding = layer.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )

        # Attention flow
        self.attention = AttentionFlow(2 * hidden_size, device=device)

        # Modeling layer
        self.modeling_layer = layer.LSTM(
            8 * hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=self.dropout_rate,
        )

        # Output layer
        self.out_lstm = layer.LSTM(
            2 * hidden_size, hidden_size, batch_first=True, bidirectional=True,
        )
        self.output_layer = layer.QAOutput(
            10 * hidden_size,
            1,
            dropout_rate=self.dropout_rate,
            classifier_bias=False,
            device=device,
        )

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def forward(self, **inputs):
        # Extract masks and lenghts from the inputs
        questions_mask = inputs["question_attention_mask"]
        contexts_mask = inputs["context_attention_mask"]
        questions_lenght = inputs["question_lenghts"]
        contexts_lenght = inputs["context_lenghts"]

        # Embed questions and contexts separately
        embedded_questions = self.dropout(self.word_embedding(inputs["question_ids"]))
        embedded_contexts = self.dropout(self.word_embedding(inputs["context_ids"]))

        # Project questions and contexts into a lower-dimensional space
        hidden_questions = self.projection(embedded_questions)
        hidden_contexts = self.projection(embedded_contexts)

        # Call the highway network for questions and contexts separately
        highway_questions = self.highway(hidden_questions)
        highway_contexts = self.highway(hidden_contexts)

        # Contextual embedding layer
        contextual_questions, _ = self.contextual_embedding(
            highway_questions, questions_lenght
        )
        contextual_contexts, _ = self.contextual_embedding(
            highway_contexts, contexts_lenght
        )

        # Attention flow
        query_aware_contexts = self.attention(
            contextual_questions, contextual_contexts, questions_mask, contexts_mask
        )

        # Modeling layer and output layer
        modeling, _ = self.modeling_layer(query_aware_contexts, contexts_lenght)
        m2, _ = self.out_lstm(modeling, contexts_lenght)
        return self.output_layer(
            torch.cat([query_aware_contexts, modeling], dim=-1),
            torch.cat([query_aware_contexts, m2], dim=-1),
            **inputs,
        )


class QABertModel(QAModel):
    """
    BERT model wrapper, for question answering tasks
    
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova
    """

    BERT_OUTPUT_SIZE = 768
    MODEL_TYPE = "bert-base-uncased"

    def __init__(self, dropout_rate=0.2, device="cpu"):
        super().__init__()

        # BERT model
        self.bert_model = self.get_model()

        # Output layer
        self.out_lstm = layer.LSTM(
            self.BERT_OUTPUT_SIZE,
            self.BERT_OUTPUT_SIZE,
            batch_first=True,
            bidirectional=False,
        )
        self.output_layer = layer.QAOutput(
            self.BERT_OUTPUT_SIZE,
            1,
            dropout_rate=dropout_rate,
            classifier_bias=False,
            device=device,
        )

        # Transfer model to device
        self.device = device
        self.to(self.device)

    def get_model(self):
        """
        Returns a pre-trained BERT model
        """
        return transformers.BertModel.from_pretrained(self.MODEL_TYPE)

    def get_model_inputs(self, **inputs):
        """
        Return a subset of the given input dict, 
        to be used as inputs for the wrapped Transformer
        (BERT takes "input_ids", "token_type_ids" and "attention_mask" as inputs)
        """
        return {
            "input_ids": inputs["context_ids"],
            "token_type_ids": inputs["context_type_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def forward(self, **inputs):
        bert_inputs = self.get_model_inputs(**inputs)
        bert_outputs = self.bert_model(**bert_inputs)[0]
        end_input, _ = self.out_lstm(
            bert_outputs,
            torch.tensor(bert_outputs.shape[1], device=self.device).repeat(
                bert_outputs.shape[0]
            ),
        )
        outputs = self.output_layer(bert_outputs, end_input, **inputs)
        return outputs


class QADistilBertModel(QABertModel):
    """
    DistilBERT model wrapper, for question answering tasks
    
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf
    """

    MODEL_TYPE = "distilbert-base-uncased"

    def __init__(self, dropout_rate=0.2, device="cpu"):
        super().__init__(dropout_rate=dropout_rate, device=device)

    def get_model(self):
        """
        Returns a pre-trained DistilBERT model
        """
        return transformers.DistilBertModel.from_pretrained(self.MODEL_TYPE)

    def get_model_inputs(self, **inputs):
        """
        Return a subset of the given input dict, 
        to be used as inputs for the wrapped Transformer
        (DistilBERT takes "input_ids" and "attention_mask" as inputs)
        """
        return {
            "input_ids": inputs["context_ids"],
            "attention_mask": inputs["attention_mask"],
        }


class QAElectraModel(QABertModel):
    """
    ELECTRA model wrapper, for question answering tasks
    
    "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
    Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher Manning
    """

    MODEL_TYPE = "google/electra-base-discriminator"

    def __init__(self, dropout_rate=0.2, device="cpu"):
        super().__init__(dropout_rate=dropout_rate, device=device)

    def get_model(self):
        """
        Returns a pre-trained ELECTRA model
        """
        return transformers.ElectraModel.from_pretrained(self.MODEL_TYPE)
