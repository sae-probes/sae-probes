import warnings
from pathlib import Path
from typing import NamedTuple

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning

from sae_probes.constants import Setting
from sae_probes.utils_data import (
    get_classimabalance_num_train,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_xy_traintest,
    get_xy_traintest_specify,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)


class Activations(NamedTuple):
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor


# Common variables
DATASET_SIZES = get_dataset_sizes()
DATASESTS = get_numbered_binary_tags()


# Helper functions for all settings
def save_activations(path, activation):
    """Save activations in sparse format to save space"""
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)


def load_activations(path):
    """Load activations from sparse format"""
    return torch.load(path, weights_only=True).to_dense().float()


@torch.inference_mode()
def generate_sae_activations_normal(
    sae: SAE,
    dataset: str,
    hook_name: str,
    model_name: str,
    device: str,
    model_cache_path: str | Path,
    batch_size: int = 128,
) -> Activations:
    size = DATASET_SIZES[dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train,
        dataset,
        hook_name,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )

    X_train_sae = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i : i + batch_size].to(device)
        X_train_sae.append(sae.encode(batch).cpu())
    X_train_sae = torch.cat(X_train_sae)

    X_test_sae = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size].to(device)
        X_test_sae.append(sae.encode(batch).cpu())
    X_test_sae = torch.cat(X_test_sae)

    return Activations(
        X_train=X_train_sae,
        X_test=X_test_sae,
        y_train=torch.tensor(y_train),
        y_test=torch.tensor(y_test),
    )


@torch.inference_mode()
def generate_sae_activations_scarcity(
    sae: SAE,
    dataset: str,
    hook_name: str,
    model_name: str,
    device: str,
    num_train: int,
    model_cache_path: str | Path,
    batch_size: int = 128,
) -> Activations:
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train,
        dataset,
        hook_name,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )

    X_train_sae = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i : i + batch_size].to(device)
        X_train_sae.append(sae.encode(batch).cpu())
    X_train_sae = torch.cat(X_train_sae)

    X_test_sae = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size].to(device)
        X_test_sae.append(sae.encode(batch).cpu())
    X_test_sae = torch.cat(X_test_sae)

    return Activations(
        X_train=X_train_sae,
        X_test=X_test_sae,
        y_train=torch.tensor(y_train),
        y_test=torch.tensor(y_test),
    )


@torch.inference_mode()
def generate_sae_activations_imbalance(
    sae: SAE,
    dataset: str,
    hook_name: str,
    model_name: str,
    device: str,
    frac: float,
    model_cache_path: str | Path,
    batch_size: int = 128,
) -> Activations:
    """Generate and save SAE activations for class imbalance setting"""
    num_train, num_test = get_classimabalance_num_train(dataset)
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        dataset,
        hook_name,
        pos_ratio=frac,
        model_name=model_name,
        num_test=num_test,
        model_cache_path=model_cache_path,
    )

    X_train_sae = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i : i + batch_size].to(device)
        X_train_sae.append(sae.encode(batch).cpu())
    X_train_sae = torch.cat(X_train_sae)

    X_test_sae = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size].to(device)
        X_test_sae.append(sae.encode(batch).cpu())
    X_test_sae = torch.cat(X_test_sae)

    return Activations(
        X_train=X_train_sae,
        X_test=X_test_sae,
        y_train=torch.tensor(y_train),
        y_test=torch.tensor(y_test),
    )


def generate_sae_activations(
    sae: SAE,
    setting: Setting,
    dataset: str,
    hook_name: str,
    model_name: str,
    device: str,
    num_train: int | None,
    frac: float | None,
    model_cache_path: str | Path,
    batch_size: int = 128,
) -> Activations:
    if setting == "normal":
        return generate_sae_activations_normal(
            sae=sae,
            dataset=dataset,
            hook_name=hook_name,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_cache_path=model_cache_path,
        )
    elif setting == "scarcity":
        assert num_train is not None
        return generate_sae_activations_scarcity(
            sae=sae,
            dataset=dataset,
            hook_name=hook_name,
            model_name=model_name,
            device=device,
            num_train=num_train,
            batch_size=batch_size,
            model_cache_path=model_cache_path,
        )
    elif setting == "imbalance":
        assert frac is not None
        return generate_sae_activations_imbalance(
            sae=sae,
            dataset=dataset,
            hook_name=hook_name,
            model_name=model_name,
            device=device,
            frac=frac,
            batch_size=batch_size,
            model_cache_path=model_cache_path,
        )
    else:
        raise ValueError(f"Invalid setting: {setting}")
