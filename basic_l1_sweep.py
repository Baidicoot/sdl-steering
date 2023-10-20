import training.dictionary as sae

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="activation_data")
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

