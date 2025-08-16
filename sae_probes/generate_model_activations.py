import glob
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sae_probes.constants import DATA_PATH
from sae_probes.utils_hooks import get_layer_from_hook_name


def _get_tokenizer(model: HookedTransformer) -> PreTrainedTokenizerBase:
    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokenizer.truncation_side = "left"  # type: ignore
    tokenizer.padding_side = "right"  # type: ignore
    return tokenizer


def _get_text_lengths(model: HookedTransformer, texts: list[str]) -> list[int]:
    tokenizer = _get_tokenizer(model)
    return [len(tokenizer(t)["input_ids"]) for t in texts]  # type: ignore


@torch.inference_mode()
def _process_activations(
    model: HookedTransformer,
    texts: list[str],
    batch_size: int,
    max_seq_len: int,
    hook_names: list[str],
    max_layer: int | None,
    device: str,
) -> dict[str, torch.Tensor]:
    tokenizer = _get_tokenizer(model)
    text_lengths = _get_text_lengths(model, texts)
    all_activations = {hook_name: [] for hook_name in hook_names}
    bar = tqdm(range(0, len(texts), batch_size))
    for i in bar:
        batch_text = texts[i : i + batch_size]
        batch_lengths = text_lengths[i : i + batch_size]
        batch = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )  # type: ignore
        batch = batch.to(device)
        # Determine a safe stop layer if not provided: use the maximum blocks.{i}.*
        if max_layer is None:
            max_block: int = -1
            for name in hook_names:
                layer_idx = get_layer_from_hook_name(name)
                if layer_idx is not None and layer_idx > max_block:
                    max_block = layer_idx
            computed_stop_at_layer = (max_block + 1) if max_block >= 0 else None
        else:
            computed_stop_at_layer = max_layer + 1

        if computed_stop_at_layer is not None:
            _, cache = model.run_with_cache(
                batch["input_ids"],
                names_filter=hook_names,
                stop_at_layer=computed_stop_at_layer,
            )
        else:
            _, cache = model.run_with_cache(
                batch["input_ids"],
                names_filter=hook_names,
            )
        for j, length in enumerate(batch_lengths):
            for hook_name in hook_names:
                activation_pos = min(length - 1, max_seq_len - 1)
                all_activations[hook_name].append(
                    cache[hook_name][j, activation_pos].cpu()
                )
        bar.set_description(f"{len(all_activations[hook_name])}")

    return {
        hook_name: torch.stack(activations)
        for hook_name, activations in all_activations.items()
    }


@torch.inference_mode()
def generate_single_dataset_activations(
    model: HookedTransformer,
    model_name: str,
    dataset_path: str | Path,
    hook_names: list[str],
    model_cache_path: str | Path,
    device: str = "cuda",
    max_seq_len: int = 1024,
    batch_size: int = 32,
    OOD: bool = False,
):
    dataset = pd.read_csv(dataset_path, compression="zstd")
    if "prompt" not in dataset.columns:
        return
    dataset_short_name = str(dataset_path).split("/")[-1].split(".")[0]
    file_names = [
        Path(model_cache_path)
        / f"model_activations_{model_name}{'_OOD' if OOD else ''}"
        / f"{dataset_short_name}_{hook_name}.pt"
        for hook_name in hook_names
    ]
    lengths = None
    if all(os.path.exists(file_name) for file_name in file_names):
        lengths = [
            torch.load(file_name, weights_only=True).shape[0]
            for file_name in file_names
        ]

    text = dataset["prompt"].tolist()
    text_lengths = _get_text_lengths(model, text)

    if lengths is not None and all(length == len(text_lengths) for length in lengths):
        print(
            f"Skipping {dataset_short_name} because correct length activations already exist"
        )
        return

    if lengths is None:
        print(
            f"Generating activations for {dataset_short_name} (no existing activations)"
        )
    else:
        print(
            f"Generating activations for {dataset_short_name} (bad existing activations)"
        )
        print(lengths, len(text_lengths))

    all_activations = _process_activations(
        model=model,
        texts=text,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hook_names=hook_names,
        max_layer=None,
        device=device,
    )

    for hook_name, file_name in zip(hook_names, file_names):
        file_name.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_activations[hook_name], file_name)


@torch.inference_mode()
def generate_dataset_activations(
    model_name: str,
    hook_names: list[str],
    model_cache_path: str | Path,
    device: str = "cuda",
    max_seq_len: int = 1024,
    batch_size: int = 32,
    OOD: bool = False,
    model: HookedTransformer | None = None,
):
    os.makedirs(
        Path(model_cache_path)
        / f"model_activations_{model_name}{'_OOD' if OOD else ''}",
        exist_ok=True,
    )

    # Load the model
    if model is None:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=device
        )

    if OOD:
        dataset_paths = glob.glob(str(DATA_PATH / "OOD data" / "*.csv"))
    else:
        dataset_paths = glob.glob(str(DATA_PATH / "cleaned_data" / "*.csv"))

    for dataset_path in dataset_paths:
        generate_single_dataset_activations(
            model=model,
            model_name=model_name,
            dataset_path=dataset_path,
            hook_names=hook_names,
            model_cache_path=model_cache_path,
            device=device,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            OOD=OOD,
        )


@torch.inference_mode()
def ensure_dataset_activations(
    model_name: str,
    dataset_short_names: list[str],
    hook_names: list[str],
    model_cache_path: str | Path,
    device: str = "cuda",
    max_seq_len: int = 1024,
    batch_size: int = 32,
    OOD: bool = False,
    model: HookedTransformer | None = None,
) -> None:
    """Ensure activations are present for each dataset/hook pair; generate missing ones."""
    to_generate: list[tuple[str, str]] = []
    base_dir = (
        Path(model_cache_path)
        / f"model_activations_{model_name}{'_OOD' if OOD else ''}"
    )
    for dataset in dataset_short_names:
        for hook in hook_names:
            expected = base_dir / f"{dataset}{'_OOD' if OOD else ''}_{hook}.pt"
            if not expected.exists():
                to_generate.append((dataset, hook))

    if not to_generate:
        return

    # Load model once if needed
    if model is None:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=device
        )

    # Generate per dataset for all requested hooks
    for dataset in sorted({d for d, _ in to_generate}):
        dataset_path = (
            DATA_PATH
            / ("OOD data" if OOD else "cleaned_data")
            / f"{dataset}{'_OOD' if OOD else ''}.csv.zst"
        )
        hooks_for_dataset = [h for d, h in to_generate if d == dataset]
        generate_single_dataset_activations(
            model=model,
            model_name=model_name,
            dataset_path=dataset_path,
            hook_names=hooks_for_dataset,
            model_cache_path=model_cache_path,
            device=device,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            OOD=OOD,
        )
