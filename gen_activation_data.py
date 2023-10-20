import datagen.sample_activations
from transformers import AutoModelForCausalLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--skip_chunks", type=int, default=0)
    parser.add_argument("--chunk_size_acts", type=int, default=64 * 1024)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layers", type=int, nargs="+", default=[2])
    parser.add_argument("--locations", type=str, nargs="+", default=[])
    parser.add_argument("--dataset_folder", type=str, default="activation_data")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if len(args.locations) == 0:
        # print modules of the model
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

        print("No hook locations specified, possible locations are:")

        for name, _ in model.named_modules():
            print(name)
        
        exit(0)

    datagen.sample_activations.setup_data(
        args.model,
        args.dataset,
        args.dataset_folder,
        args.locations,
        args.chunk_size_acts,
        args.n_chunks,
        skip_chunks=args.skip_chunks,
        device=args.device,
    )