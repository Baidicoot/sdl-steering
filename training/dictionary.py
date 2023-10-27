import torch
import torch.nn as nn

from torchtyping import TensorType

import copy
from typing import Literal

from torch.func import stack_module_state, vmap, functional_call

import torchopt

import training.ensemble as ens

class SparseLinearAutoencoder(nn.Module):
    def __init__(
        self,
        activation_size,
        n_dict_components,
        l1_alpha,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.encoder = nn.Parameter(torch.empty((n_dict_components, activation_size), device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.encoder)

        self.encoder_bias = nn.Parameter(torch.empty((n_dict_components,), device=device, dtype=dtype))
        nn.init.zeros_(self.encoder_bias)

        self.register_buffer("l1_penalty", torch.tensor(l1_alpha, device=device, dtype=dtype))
    
    def forward(self, batch):
        decoder_norms = torch.norm(self.encoder, 2, dim=-1)
        learned_dict = self.encoder / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, batch)
        c = c + self.encoder_bias[None, :]
        c = torch.clamp(c, min=0.0)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = self.l1_penalty * torch.norm(c, 1, dim=-1).mean()
        
        return l_reconstruction + l_l1, l_reconstruction, c, x_hat

def make_ensemble(input_dim, hidden_dim, l1_range, adam_settings, activation="relu", device="cuda"):
    # create a list of models
    models = []
    for l1_penalty in l1_range:
        models.append(SparseLinearAutoencoder(input_dim, hidden_dim, l1_penalty).to(device))

    ensemble = ens.Ensemble(
        models,
        optimizer_func=torchopt.adam,
        optimizer_kwargs=adam_settings,
        model_hyperparams={"input_dim": input_dim, "hidden_dim": hidden_dim, "activation_type": activation, "l1_penalty": 0},
    )

    return ensemble