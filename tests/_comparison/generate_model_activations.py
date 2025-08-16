# %%

import argparse
import glob
import os
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)


def generate_dataset_activations(
    model_name, device="cuda:0", max_seq_len=1024, OOD=False
):
    os.makedirs(
        f"data/model_activations_{model_name}{'_OOD' if OOD else ''}", exist_ok=True
    )

    # Load the model
    if model_name == "gemma-2-9b":
        model = HookedTransformer.from_pretrained("google/gemma-2-9b", device=device)
    elif model_name == "llama-3.1-8b":
        model = HookedTransformer.from_pretrained(
            "meta-llama/Llama-3.1-8B", device=device
        )
    elif model_name == "gemma-2-2b":
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Important to ensure correct token is at the correct position, either at the text_length position or at the end of the sequence
    tokenizer = model.tokenizer
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    # Define hook names based on model
    if model_name == "gemma-2-9b":
        hook_names = ["hook_embed"] + [
            f"blocks.{layer}.hook_resid_post" for layer in [9, 20, 31, 41]
        ]
    elif model_name == "llama-3.1-8b":
        hook_names = ["hook_embed"] + [
            f"blocks.{layer}.hook_resid_post" for layer in [8, 16, 24, 31]
        ]
    elif model_name == "gemma-2-2b":
        hook_names = ["hook_embed"] + [
            f"blocks.{layer}.hook_resid_post" for layer in [12]
        ]
    else:
        raise ValueError(f"Model {model_name} not supported")

    if OOD:
        dataset_names = glob.glob("data/OOD data/*.csv")
    else:
        dataset_names = glob.glob("data/cleaned_data/*.csv")

    # Randomize dataset names so multiple GPUs can work on it
    random.shuffle(dataset_names)

    for dataset_name in dataset_names:
        dataset = pd.read_csv(dataset_name)
        if "prompt" not in dataset.columns:
            continue
        dataset_short_name = dataset_name.split("/")[-1].split(".")[0]
        file_names = [
            f"data/model_activations_{model_name}{'_OOD' if OOD else ''}/{dataset_short_name}_{hook_name}.pt"
            for hook_name in hook_names
        ]
        lengths = None
        if all(os.path.exists(file_name) for file_name in file_names):
            lengths = [
                torch.load(file_name, weights_only=True).shape[0]
                for file_name in file_names
            ]

        text = dataset["prompt"].tolist()

        text_lengths = []
        for t in text:
            text_lengths.append(len(tokenizer(t)["input_ids"]))

        if lengths is not None and all(
            length == len(text_lengths) for length in lengths
        ):
            print(
                f"Skipping {dataset_short_name} because correct length activations already exist"
            )
            continue

        if lengths is not None:
            print(
                f"Generating activations for {dataset_short_name} (bad existing activations)"
            )
            print(lengths, len(text_lengths))
        else:
            print(
                f"Generating activations for {dataset_short_name} (no existing activations)"
            )

        batch_size = 1
        all_activations = {hook_name: [] for hook_name in hook_names}
        bar = tqdm(range(0, len(text), batch_size))
        for i in bar:
            batch_text = text[i : i + batch_size]
            batch_lengths = text_lengths[i : i + batch_size]
            batch = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            batch = batch.to(device)
            logits, cache = model.run_with_cache(
                batch["input_ids"], names_filter=hook_names
            )
            for j, length in enumerate(batch_lengths):
                for hook_name in hook_names:
                    activation_pos = min(length - 1, max_seq_len - 1)
                    all_activations[hook_name].append(
                        cache[hook_name][:, activation_pos].cpu()
                    )
            bar.set_description(f"{len(all_activations[hook_name])}")

        print(
            i,
            len(all_activations[hook_name]),
            len(torch.cat(all_activations[hook_name])),
        )

        for hook_name, file_name in zip(hook_names, file_names):
            all_activations[hook_name] = torch.cat(all_activations[hook_name])
            torch.save(all_activations[hook_name], file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--OOD", action="store_true")
    args = parser.parse_args()

    if args.model_name:
        # Run for a single model
        generate_dataset_activations(
            args.model_name, args.device, args.max_seq_len, args.OOD
        )
    else:
        # Run for all models
        model_names = ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]
        for model_name in model_names:
            print(f"Processing model: {model_name}")
            generate_dataset_activations(
                model_name, args.device, args.max_seq_len, args.OOD
            )
# %%
