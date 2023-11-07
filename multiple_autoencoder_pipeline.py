from jaxtyping import Int
from typing import List
import torch
from torch import Tensor, device, set_grad_enabled
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.base_store import ActivationStore, ActivationStoreItem
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.generate_activations import generate_activations
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime

from sparse_autoencoder.autoencoder.loss import (
    l1_loss,
    reconstruction_loss,
    sae_training_loss,
)

import wandb

def train_autoencoders(
    activations_dataloader: DataLoader[ActivationStoreItem],
    autoencoders: List[SparseAutoencoder],
    optimizers: List[Optimizer],
    all_sweep_parameters: List[SweepParametersRuntime],
    log_interval: int = 10,
    device: device | None = None,
) -> None:
    """Sparse Autoencoder Training Loop. Modded to train multiple autoencoders in a single run.

    Args:
        activations_dataloader: DataLoader containing activations.
        autoencoder: Sparse autoencoder model.
        optimizer: The optimizer to use.
        sweep_parameters: The sweep parameters to use.
        log_interval: How often to log progress.
        device: Decide to use.
    """
    n_dataset_items: int = len(activations_dataloader.dataset)  # type: ignore
    batch_size: int = activations_dataloader.batch_size  # type: ignore

    with set_grad_enabled(True), tqdm(  # noqa: FBT003
        desc="Train Autoencoder",
        total=n_dataset_items,
        colour="green",
        position=1,
        leave=False,
        dynamic_ncols=True,
    ) as progress_bar:
        for step, batch in enumerate(activations_dataloader):
            
            reconstruction_losses_mse = []
            l1_losses_learned_activations = []
            total_losses = []
            sparsities = []
            fvus = []

            for autoencoder, optimizer, sweep_parameters in zip(
                autoencoders, optimizers, all_sweep_parameters
            ):
                # Zero the gradients
                optimizer.zero_grad()

                # Move the batch to the device (in place)
                batch = batch.to(device)  # noqa: PLW2901

                # Forward pass
                with torch.autocast(device_type="cpu" if device is None else "cuda", dtype=torch.float16):
                    learned_activations, reconstructed_activations = autoencoder(batch)

                    # Get metrics
                    reconstruction_loss_mse = reconstruction_loss(
                        batch,
                        reconstructed_activations,
                    )
                    l1_loss_learned_activations = l1_loss(learned_activations)
                    total_loss = sae_training_loss(
                        reconstruction_loss_mse,
                        l1_loss_learned_activations,
                        sweep_parameters.l1_coefficient,
                    )

                    sparsity = learned_activations.count_nonzero(dim=-1).float().mean()
                    fvu = reconstruction_loss_mse / (batch - batch.mean(dim=0)).pow(2).mean()

                reconstruction_losses_mse.append(reconstruction_loss_mse.detach())
                l1_losses_learned_activations.append(l1_loss_learned_activations.detach())
                total_losses.append(total_loss.detach())
                sparsities.append(sparsity.detach())
                fvus.append(fvu.detach())

                # TODO: Store the learned activations (default every 25k steps)

                # Backwards pass
                total_loss.backward()

                optimizer.step()

            # Log
            if step % log_interval == 0 and wandb.run is not None:
                wandb.log(
                    {
                        "reconstruction_loss": {
                            sweep_parameters.l1_coefficient: mse
                            for sweep_parameters, mse in zip(all_sweep_parameters, reconstruction_losses_mse)
                        },
                        "l1_loss": {
                            sweep_parameters.l1_coefficient: l1
                            for sweep_parameters, l1 in zip(all_sweep_parameters, l1_losses_learned_activations)
                        },
                        "total_loss": {
                            sweep_parameters.l1_coefficient: total
                            for sweep_parameters, total in zip(all_sweep_parameters, total_losses)
                        },
                        "sparsity": {
                            sweep_parameters.l1_coefficient: sparsity
                            for sweep_parameters, sparsity in zip(all_sweep_parameters, sparsities)
                        },
                        "fvu": {
                            sweep_parameters.l1_coefficient: fvu
                            for sweep_parameters, fvu in zip(all_sweep_parameters, fvus)
                        },
                    },
                )

            # TODO: Get the feature density & also log to wandb

            # TODO: Apply neuron resampling if enabled

            progress_bar.update(batch_size)

        progress_bar.close()

def multiple_autoencoder_pipeline(
    src_model: HookedTransformer,
    src_model_activation_hook_point: str,
    src_model_activation_layer: int,
    src_dataloader: DataLoader[Int[Tensor, " pos"]],
    activation_store: ActivationStore,
    num_activations_before_training: int,
    autoencoders: List[SparseAutoencoder],
    all_sweep_parameters: List[SweepParametersRuntime],  # noqa: B008
    save_format: str = "models/{data}_{epoch}.pt",
    device: torch.device | None = None,
) -> None:
    """
    Modded pipeline for training multiple autoencoders in a single run.

    The pipeline alternates between generating activations and training the autoencoder.

    Args:
        src_model: The model to get activations from.
        src_model_activation_hook_point: The hook point to get activations from.
        src_model_activation_layer: The layer to get activations from. This is used to stop the
            model after this layer, as we don't need the final logits.
        src_dataloader: DataLoader containing source model inputs (typically batches of prompts)
            that are used to generate the activations data.
        activation_store: The store to buffer activations in once generated, before training the
            autoencoder.
        num_activations_before_training: The number of activations to generate before training the
            autoencoder. As a guide, 1 million activations, each of size 1024, will take up about
            2GB of memory (assuming float16/bfloat16).
        autoencoder: The autoencoder to train.
        sweep_parameters: Parameter config to use.
        save_format: The format to save the autoencoder in. This is a format string, where index is the index of the model.
        device: Device to run pipeline on.
    """
    for autoencoder in autoencoders:
        autoencoder.to(device)

    optimizers = []

    for autoencoder, sweep_parameters in zip(autoencoders, all_sweep_parameters):
        optimizer: Optimizer = Adam(
            autoencoder.parameters(),
            lr=sweep_parameters.lr,
            betas=(sweep_parameters.adam_beta_1, sweep_parameters.adam_beta_2),
            eps=sweep_parameters.adam_epsilon,
            weight_decay=sweep_parameters.adam_weight_decay,
        )

        optimizers.append(optimizer)

    # Run loop until source data is exhausted:
    with logging_redirect_tqdm(), tqdm(
        desc="Generate/Train Cycles",
        position=0,
        dynamic_ncols=True,
    ) as progress_bar:
        epoch = 0

        while True:
            # Add activations to the store
            generate_activations(
                src_model,
                src_model_activation_layer,
                src_model_activation_hook_point,
                activation_store,
                src_dataloader,
                device=device,
                num_items=num_activations_before_training,
            )
            if len(activation_store) == 0:
                break

            # Shuffle the store if it has a shuffle method - it is often more efficient to
            # create a shuffle method ourselves rather than get the DataLoader to shuffle
            activation_store.shuffle()

            # Create a dataloader from the store
            dataloader = DataLoader(
                activation_store,
                batch_size=sweep_parameters.batch_size,
            )

            # Train the autoencoders
            train_autoencoders(
                dataloader,
                autoencoders,
                optimizers,
                all_sweep_parameters,
                device=device,
            )

            # Save the autoencoder

            torch.save(
                autoencoders,
                save_format.format(epoch=epoch, data="autoencoders"),
            )

            # Empty the store so we can fill it up again
            activation_store.empty()

            progress_bar.update(1)
            epoch += 1