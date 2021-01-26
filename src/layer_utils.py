import torch
import torch.nn as nn

import layer


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


def get_highway(
    num_layers, size, nonlinearity=nn.ReLU, gate_activation=nn.Sigmoid, device="cpu"
):
    """
    Return a PyTorch Sequential composed of the given number
    of Highway modules
    """
    highway = []
    for _ in range(num_layers):
        highway.append(
            layer.Highway(
                size,
                nonlinearity=nonlinearity,
                gate_activation=gate_activation,
                device=device,
            )
        )
    return nn.Sequential(*highway)
