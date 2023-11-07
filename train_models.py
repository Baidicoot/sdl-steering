from sparse_autoencoder import SparseAutoencoder, pipeline
from sparse_autoencoder.source_data.pile_uncopyrighted import PileUncopyrightedDataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from transformers import PreTrainedTokenizerBase
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime
import torch
import wandb
import json
import datetime
import numpy as np
import os
from typing import List, Tuple

from multiple_autoencoder_pipeline import multiple_autoencoder_pipeline
from tensor_store import TensorActivationStore

device = get_device()

src_model = HookedTransformer.from_pretrained("gpt2-small", dtype="float32")
src_d_mlp: int = src_model.cfg.d_model # type: ignore

tokenizer: PreTrainedTokenizerBase = src_model.tokenizer  # type: ignore
source_data = PileUncopyrightedDataset(tokenizer=tokenizer)
src_dataloader = source_data.get_dataloader(batch_size=8)

max_items = 1_000_000
store = TensorActivationStore(max_items, src_d_mlp, device)

def initialize_autoencoders(
        l1_range: List[float]
) -> Tuple[List[SparseAutoencoder], List[SweepParametersRuntime]]:
    autoencoders = []
    all_sweep_parameters = []

    for l1 in l1_range:
        autoencoder = SparseAutoencoder(src_d_mlp, src_d_mlp * 8, torch.zeros(src_d_mlp))
        sweep_parameters = SweepParametersRuntime(
            l1_coefficient=l1,
        )

        autoencoders.append(autoencoder)
        all_sweep_parameters.append(sweep_parameters)
    
    return autoencoders, all_sweep_parameters

autoencoders, all_sweep_parameters = initialize_autoencoders(list(np.logspace(-4, -2, 10)))

os.makedirs("models", exist_ok=True)

with open("secrets/wandb_cfg.json", "r") as f:
    wandb_cfg = json.load(f)
wandb.login(key=wandb_cfg["api_key"])

now = datetime.datetime.now()
timestr = now.strftime("%Y-%m-%d_%H-%M")

wandb.init(
    project=wandb_cfg["project"],
    entity=wandb_cfg["entity"],
    name=f"{wandb_cfg['run_name']}_{timestr}"
)

multiple_autoencoder_pipeline(
    src_model=src_model,
    src_model_activation_hook_point="blocks.9.attn.hook_z",
    src_model_activation_layer=9,
    src_dataloader=src_dataloader,
    activation_store=store,
    num_activations_before_training=max_items,
    autoencoders=autoencoders,
    all_sweep_parameters=all_sweep_parameters,
    device=device,
)