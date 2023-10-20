import argparse
import importlib
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from datasets import Dataset, DatasetDict, load_dataset
from einops import rearrange
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, AutoTokenizer

T = TypeVar("T", bound=Union[Dataset, DatasetDict])

def read_from_pile(address: str, max_lines: int = 100_000, start_line: int = 0):
    """Reads a file from the Pile dataset. Returns a generator."""

    with open(address, "r") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= max_lines + start_line:
                break
            yield json.loads(line)


def make_sentence_dataset(dataset_name: str, max_lines: int = 20_000, start_line: int = 0):
    """Returns a dataset from the Huggingface Datasets library."""
    if dataset_name == "EleutherAI/pile":
        if not os.path.exists("pile0"):
            print("Downloading shard 0 of the Pile dataset (requires 50GB of disk space).")
            if not os.path.exists("pile0.zst"):
                os.system("curl https://the-eye.eu/public/AI/pile/train/00.jsonl.zst > pile0.zst")
                os.system("unzstd pile0.zst")
        dataset = Dataset.from_list(list(read_from_pile("pile0", max_lines=max_lines, start_line=start_line)))
    else:
        dataset = load_dataset(dataset_name, split="train")#, split=f"train[{start_line}:{start_line + max_lines}]")
    return dataset


# Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = min(mp.cpu_count() // 2, 8),
    text_key: str = "text",
    max_length: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> Tuple[T, float]:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_length: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_fn(x: Dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_length)  # tokenizer max length is 1024 for gpt2
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size] for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return data.with_format(format, columns=["input_ids"]), (total_tokens / total_bytes) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


# End Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py

def make_activation_dataset_hf(
    sentence_dataset: Dataset,
    model: AutoModelForCausalLM,
    tensor_names: List[str],
    chunk_size: int,
    n_chunks: int,
    output_folder: str = "activation_data",
    skip_chunks: int = 0,
    device: Optional[torch.device] = torch.device("cuda:0"),
    max_length: int = 2048,
    model_batch_size: int = 4,
    precision: Literal["float16", "float32"] = "float16",
    shuffle_seed: Optional[int] = None,
):
    with torch.no_grad():
        try:
            os.makedirs(output_folder, exist_ok=False)
        except FileExistsError:
            print(f"Output folder '{output_folder}' already exists, skipping...")
            return

        model.eval()

        dtype = None
        if precision == "float16":
            dtype = torch.float16
        elif precision == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{precision}'")

        chunk_batches = chunk_size // (model_batch_size * max_length)
        batches_to_skip = skip_chunks * chunk_batches

        if shuffle_seed is not None:
            torch.manual_seed(shuffle_seed)

        dataloader = DataLoader(
            sentence_dataset,
            batch_size=model_batch_size,
            shuffle=shuffle_seed is not None,
        )

        dataloader_iter = iter(dataloader)

        for _ in range(batches_to_skip):
            dataloader_iter.__next__()
        
        # configure hooks for the model
        tensor_buffer: Dict[str, Any] = {}

        hook_handles = []

        for tensor_name in tensor_names:
            tensor_buffer[tensor_name] = []

            print(tensor_name)

            os.makedirs(os.path.join(output_folder, tensor_name))

            def hook(module, input, output, tensor_name=tensor_name):
                if type(output) == tuple:
                    out = output[0]
                else:
                    out = output
                if not isinstance(tensor_name, str):
                    raise ValueError(f"Tensor name must be a string")
                tensor_buffer[tensor_name].append(rearrange(out, "b l ... -> (b l) (...)").to(dtype=dtype).cpu())
                return output

            name_was_set = False
            for name, module in model.named_modules():
                if name == tensor_name:
                    handle = module.register_forward_hook(hook)
                    hook_handles.append(handle)
                    name_was_set = True
            
            if not name_was_set:
                raise ValueError(f"Tensor name '{tensor_name}' not found in model")

        def reset_buffers():
            for tensor_name in tensor_names:
                tensor_buffer[tensor_name] = []

        reset_buffers()

        chunk_idx = 0

        progress_bar = tqdm(total=chunk_size * n_chunks)

        for batch_idx, batch in enumerate(dataloader_iter):
            batch = batch["input_ids"].to(device)

            _ = model(batch)

            if batch_idx == 0:
                # save tensor sizes and generation config to disk to output_folder/gen_cfg.json
                # first check if file exists and if so, load it
                tensor_sizes: Dict[str, int] = {}
                gen_cfg_path = os.path.join(output_folder, "gen_cfg.json")
                
                for tensor_name in tensor_names:
                    tensor_sizes[tensor_name] = tensor_buffer[tensor_name][0].shape[-1]
                
                with open(gen_cfg_path, "w") as f:
                    gen_cfg = {
                        "chunk_size": chunk_size,
                        "n_chunks": n_chunks,
                        "max_length": max_length,
                        "model_batch_size": model_batch_size,
                        "precision": precision,
                        "shuffle_seed": shuffle_seed,
                        "tensor_sizes": tensor_sizes,
                    }
                    json.dump(gen_cfg, f)

            progress_bar.update(model_batch_size * max_length)

            if (batch_idx+1) % chunk_batches == 0:
                for tensor_name in tensor_names:
                    save_activation_chunk(tensor_buffer[tensor_name], chunk_idx, os.path.join(output_folder, tensor_name))
                
                n_act = batch_idx * model_batch_size * max_length
                print(f"Saved chunk {chunk_idx} of activations, total size: {n_act / 1e6:.2f}M activations")

                chunk_idx += 1
                
                reset_buffers()
                if chunk_idx >= n_chunks:
                    break
        
        # undersized final chunk
        if chunk_idx < n_chunks:
            for tensor_name in tensor_names:
                save_activation_chunk(tensor_buffer[tensor_name], chunk_idx, os.path.join(output_folder, tensor_name))
            
            n_act = batch_idx * model_batch_size * max_length
            print(f"Saved undersized chunk {chunk_idx} of activations, total size: {n_act / 1e6:.2f}M activations")

        for hook_handle in hook_handles:
            hook_handle.remove()
        

def save_activation_chunk(dataset, n_saved_chunks, dataset_folder):
    dataset_t = torch.cat(dataset, dim=0).to("cpu")
    os.makedirs(dataset_folder, exist_ok=True)
    with open(dataset_folder + "/" + str(n_saved_chunks) + ".pt", "wb") as f:
        torch.save(dataset_t, f)

def setup_data(
    model_name: str,
    dataset_name: str,
    output_folder: str,
    tensor_names: List[str],
    chunk_size: int,
    n_chunks: int,
    skip_chunks: int = 0,
    device: Optional[torch.device] = torch.device("cuda:0"),
    max_length: int = 2048,
    model_batch_size: int = 4,
    precision: Literal["float16", "float32"] = "float16",
    shuffle_seed: Optional[int] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)

    # weak upper bound on number of lines
    max_lines = int((chunk_size * (n_chunks + skip_chunks)) / max_length) * 2

    print(f"Processing first {max_lines} lines of dataset...")

    sentence_dataset = make_sentence_dataset(dataset_name, max_lines=max_lines)
    tokenized_sentence_dataset, _ = chunk_and_tokenize(sentence_dataset, tokenizer, max_length=max_length)
    make_activation_dataset_hf(
        tokenized_sentence_dataset,
        model,
        tensor_names,
        chunk_size,
        n_chunks,
        output_folder=output_folder,
        skip_chunks=skip_chunks,
        device=device,
        max_length=max_length,
        model_batch_size=model_batch_size,
        precision=precision,
        shuffle_seed=shuffle_seed,
    )