import training.dictionary as sae

import json

import numpy as np
import torch
import tqdm

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="activation_data")
    parser.add_argument("--tensor_name", type=str, default="activations")
    parser.add_argument("--blowup_ratio", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--min_l1_penalty", type=float, default=-4)
    parser.add_argument("--max_l1_penalty", type=float, default=-1)
    parser.add_argument("--l1_penalty_spacing", type=str, default="log")
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--train_unsparse_baseline", action="store_true")
    parser.add_argument("--adam_lr", type=float, default=1e-3)

    args = parser.parse_args()

    # load dataset config
    with open(f"{args.dataset_folder}/gen_cfg.json", "r") as f:
        dataset_config = json.load(f)
    
    needs_precision_cast = dataset_config["precision"] == "float16"

    activation_size = dataset_config["tensor_sizes"][args.tensor_name]
    latent_dim = activation_size * args.blowup_ratio

    if args.l1_penalty_spacing == "log":
        l1_range = np.logspace(args.min_l1_penalty, args.max_l1_penalty, args.n_models)
    elif args.l1_penalty_spacing == "linear":
        l1_range = np.linspace(args.min_l1_penalty, args.max_l1_penalty, args.n_models)
    
    train_step, unstack_models, params, buffers = sae.make_ensemble(
        activation_size, latent_dim, l1_range, {"lr": args.adam_lr}
    )

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch}")

        chunk_idxs = np.arange(dataset_config["n_chunks"])
        np.random.shuffle(chunk_idxs)

        for chunk in chunk_idxs:
            dataset = torch.load(f"{args.dataset_folder}/{args.tensor_name}/{chunk}.pt")

            if needs_precision_cast:
                dataset = dataset.to(torch.float32)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True
            )

            for batch in tqdm.tqdm(dataloader):
                train_step(batch)
    
    models = unstack_models(params, buffers, args.device)

    torch.save(models, "models.pt")