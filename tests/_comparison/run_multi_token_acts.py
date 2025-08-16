# %%
import os

os.environ["OMP_NUM_THREADS"] = "10"

import argparse
import os
import pickle as pkl
import warnings

import einops
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils_training import find_best_reg

from .utils_data import (
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_train_test_indices,
    get_yvals,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)

data_dir = "data"
model_name = "gemma-2-9b"
max_seq_len = 256
layer = 20
k = 128
device = "cuda:0"

# Default SAE ID parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--l0", type=int, default=68, help="L0 value for the SAE", choices=[408, 68]
)
parser.add_argument(
    "--to_run_list",
    type=str,
    nargs="+",
    default=[],
    choices=["baseline_attn", "sae_aggregated", "attn_probing"],
)
args = parser.parse_args()

# Set SAE ID based on arguments
l0 = args.l0
sae_id = f"layer_20/width_16k/average_l0_{l0}"


baseline_csv = pd.read_csv(
    f"results/baseline_probes_{model_name}/normal_settings/layer{layer}_results.csv"
)
sae_csv = pd.read_csv(f"results/sae_probes_{model_name}/normal_setting/all_metrics.csv")

datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()

to_run_list = args.to_run_list
# %%


def load_model_acts(dataset):
    """Load the original model activations for a dataset"""
    hook_name = f"blocks.{layer}.hook_resid_post"
    file_path = f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_{hook_name}.pt"
    return torch.load(file_path, weights_only=True)


def load_sae_acts(dataset, sae_id):
    """Load the SAE-encoded activations for a dataset"""
    width = sae_id.split("/")[1]
    l0 = sae_id.split("/")[2]
    train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_X_train_sae.pt"
    test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_X_test_sae.pt"
    y_train_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_y_train.pt"
    y_test_path = f"{data_dir}/sae_activations_{model_name}_{max_seq_len}/{dataset}_{layer}_{width}_{l0}_y_test.pt"

    return {
        "X_train": torch.load(train_path, weights_only=True).to_dense(),
        "X_test": torch.load(test_path, weights_only=True).to_dense(),
        "y_train": torch.load(y_train_path, weights_only=True).to_dense(),
        "y_test": torch.load(y_test_path, weights_only=True).to_dense(),
    }


def train_concat_baseline_on_model_acts(
    X_train, X_test, y_train, y_test, number_to_concat=255, pca_k=20
):
    """Train baseline probe on original model activations"""
    # Get sequence length and feature dimension
    seq_len = X_train.shape[1]

    # Initialize lists to store PCA transformed data
    train_pca_features = []
    test_pca_features = []

    # For each token position
    for pos in tqdm(range(1, number_to_concat + 1)):
        if pos >= seq_len:
            break

        # Get token features for this position
        X_train_pos = X_train[:, pos, :]
        X_test_pos = X_test[:, pos, :]

        train_sums = X_train_pos.sum(dim=-1)
        if train_sums.max() == 0:
            continue

        # Fit PCA on training data
        pca = PCA(n_components=pca_k)
        X_train_pca = pca.fit_transform(X_train_pos)
        X_test_pca = pca.transform(X_test_pos)

        train_pca_features.append(X_train_pca)
        test_pca_features.append(X_test_pca)

    # Concatenate all PCA features
    X_train_concat = np.hstack(train_pca_features)
    X_test_concat = np.hstack(test_pca_features)

    res = find_best_reg(
        X_train=X_train_concat,
        y_train=y_train,
        X_test=X_test_concat,
        y_test=y_test,
        plot=False,
        n_jobs=-1,
        parallel=False,
        penalty="l1",
    )

    return res


def largest_nonzero_col_per_row(A: torch.Tensor, sentinel: int = -1) -> torch.Tensor:
    """
    Returns a 1D tensor of length A.size(0), where each entry is the
    largest column index of a nonzero element in that row of A.
    If a row is entirely zero, its index is set to `sentinel`.

    This uses the "mask * column-indices + max" approach.

    Args:
        A (torch.Tensor): A 2D tensor of shape (rows, cols).
        sentinel (int): The value to assign for rows with no nonzero entries.

    Returns:
        torch.Tensor: A 1D tensor of length A.size(0).
    """
    # 1. Create a boolean mask for nonzero entries
    mask = A != 0  # shape: [rows, cols]

    # 2. Create an integer range [0, 1, 2, ..., cols-1]
    #    and let it broadcast to [rows, cols] when multiplied by mask.
    cols = torch.arange(A.size(1), device=A.device)

    # 3. Multiply mask by the column indices and take max across each row.
    #    This effectively picks the largest column index where the row is nonzero.
    max_indices_per_row = (mask * cols).max(dim=1).values

    # 4. Identify rows that have no nonzero entries. Their max will be 0,
    #    but that may also be a valid column index. So we set them to sentinel explicitly.
    no_nonzeros = ~mask.any(dim=1)
    max_indices_per_row[no_nonzeros] = sentinel

    return max_indices_per_row


def train_aggregated_probe_on_acts(
    X_train, X_test, y_train, y_test, aggregation_method, k=None, binarize=False
):
    """Train probe on aggregated activations"""

    train_sums = X_train.sum(dim=-1)
    last_nonzero_train = largest_nonzero_col_per_row(train_sums)

    test_sums = X_test.sum(dim=-1)
    last_nonzero_test = largest_nonzero_col_per_row(test_sums)

    if aggregation_method == "mean":
        # Create masks for each sequence to only include tokens after first_nonzero, skipping first token
        train_mask = (
            torch.arange(X_train.size(1))[None, :] <= last_nonzero_train[:, None]
        ) & (torch.arange(X_train.size(1))[None, :] > 0)
        test_mask = (
            torch.arange(X_test.size(1))[None, :] <= last_nonzero_test[:, None]
        ) & (torch.arange(X_test.size(1))[None, :] > 0)

        # Apply masks and take mean only over valid tokens
        X_train_agg = (X_train * train_mask[:, :, None]).sum(dim=1) / train_mask.sum(
            dim=1
        )[:, None]
        X_test_agg = (X_test * test_mask[:, :, None]).sum(dim=1) / test_mask.sum(dim=1)[
            :, None
        ]
    elif aggregation_method == "max":
        # Set tokens before first_nonzero and first token to negative infinity so they won't be selected by max
        train_mask = (
            torch.arange(X_train.size(1))[None, :] <= last_nonzero_train[:, None]
        ) & (torch.arange(X_train.size(1))[None, :] > 0)
        test_mask = (
            torch.arange(X_test.size(1))[None, :] <= last_nonzero_test[:, None]
        ) & (torch.arange(X_test.size(1))[None, :] > 0)

        X_train_masked = X_train.clone()
        X_test_masked = X_test.clone()
        X_train_masked[~train_mask.unsqueeze(-1).expand_as(X_train)] = float("-inf")
        X_test_masked[~test_mask.unsqueeze(-1).expand_as(X_test)] = float("-inf")

        X_train_agg = X_train_masked.max(dim=1).values
        X_test_agg = X_test_masked.max(dim=1).values
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    if binarize:
        X_train_agg = (X_train_agg > 1).float()
        X_test_agg = (X_test_agg > 1).float()

    if k is not None:
        X_train_diff = X_train_agg[y_train == 1].mean(dim=0) - X_train_agg[
            y_train == 0
        ].mean(dim=0)
        sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_agg[:, top_by_average_diff]
        X_test_filtered = X_test_agg[:, top_by_average_diff]
    else:
        X_train_filtered = X_train_agg
        X_test_filtered = X_test_agg

    res = find_best_reg(
        X_train=X_train_filtered,
        y_train=y_train,
        X_test=X_test_filtered,
        y_test=y_test,
        plot=False,
        n_jobs=-1,
        parallel=False,
        penalty="l1",
    )

    return res


# %%

os.makedirs(f"data/consolidated_probing_{model_name}", exist_ok=True)


def run_sae_aggregated_probing(dataset, layer, sae_id, k, binarize=False):
    save_paths = [
        f"data/consolidated_probing_{model_name}/{dataset}_{layer}_width16k_l0{l0}_mean{'_binarized' if binarize else ''}.pkl",
        f"data/consolidated_probing_{model_name}/{dataset}_{layer}_width16k_l0{l0}_max{'_binarized' if binarize else ''}.pkl",
    ]
    all_exist = all(os.path.exists(save_path) for save_path in save_paths)
    if all_exist:
        return

    size = dataset_sizes[dataset]
    num_train = min(size - 100, 1024)
    num_test = size - num_train
    y = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio=0.5, seed=42
    )

    y_train = y[train_indices]
    y_test = y[test_indices]

    sae_acts = load_sae_acts(dataset, sae_id)

    X_train_sae = sae_acts["X_train"]
    X_test_sae = sae_acts["X_test"]

    for sae_aggregation_method, save_path in zip(["mean", "max"], save_paths):
        metrics = train_aggregated_probe_on_acts(
            X_train_sae,
            X_test_sae,
            y_train,
            y_test,
            sae_aggregation_method,
            k,
            binarize,
        )
        metrics["sae_aggregation_method"] = sae_aggregation_method
        metrics["dataset"] = dataset
        metrics["layer"] = layer
        metrics["sae_id"] = sae_id
        metrics["k"] = k

        with open(save_path, "wb") as f:
            pkl.dump(metrics, f)


binarize = True
if "sae" in to_run_list:
    for dataset in datasets:
        try:
            run_sae_aggregated_probing(dataset, layer, sae_id, k, binarize)
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            continue

# %%


def run_baseline_concat_probing(dataset, layer, sae_id, number_to_concat=255, pca_k=20):
    save_path = f"data/consolidated_probing_{model_name}/{dataset}_{layer}_baseline_{number_to_concat}_{pca_k}.pkl"
    if os.path.exists(save_path):
        return

    # Load original model activations
    model_acts = load_model_acts(dataset)

    size = dataset_sizes[dataset]
    num_train = min(size - 100, 1024)
    num_test = size - num_train
    y = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio=0.5, seed=42
    )

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train_model = model_acts[train_indices]
    X_test_model = model_acts[test_indices]

    metrics = train_concat_baseline_on_model_acts(
        X_train_model, X_test_model, y_train, y_test, number_to_concat, pca_k
    )
    metrics["dataset"] = dataset
    metrics["layer"] = layer
    metrics["number_to_concat"] = number_to_concat
    metrics["pca_k"] = pca_k

    with open(save_path, "wb") as f:
        pkl.dump(metrics, f)


number_to_concat = 255
pca_k = 20

if "baseline_attn" in to_run_list:
    for dataset in datasets:
        try:
            run_baseline_concat_probing(dataset, layer, sae_id, number_to_concat, pca_k)
        except FileNotFoundError:
            continue


# %%
def train_attn_probing(X_train, X_test, y_train, y_test, l2_lambda=0):
    """Train a simple attention-based probe on the data with learning"""
    # Get sequence length and feature dimension
    seq_len = X_train.shape[1]
    feat_dim = X_train.shape[2]

    # Convert to torch tensors if needed
    X_train = (
        torch.tensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
    )
    X_test = torch.tensor(X_test) if not isinstance(X_test, torch.Tensor) else X_test
    y_train = (
        torch.tensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
    )
    y_test = torch.tensor(y_test) if not isinstance(y_test, torch.Tensor) else y_test

    # Split train into train/val
    n_train = int(0.8 * len(X_train))
    indices = torch.randperm(len(X_train))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train_split = X_train[train_idx].to(device)
    y_train_split = y_train[train_idx].to(device)

    # Initialize learnable parameters
    output_vector = torch.nn.Parameter(torch.randn(feat_dim).to(device) * 0.001)
    attn_vector = torch.nn.Parameter(torch.randn(feat_dim).to(device) * 0.001)
    bias = torch.nn.Parameter(torch.zeros(1).to(device))
    optimizer = torch.optim.Adam([output_vector, attn_vector, bias], lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Track best validation performance
    best_val_auc = 0
    best_output_vector = None
    best_attn_vector = None
    best_bias = None

    val_vectors = X_train[val_idx].to(device)
    pbar = tqdm(range(1000))
    for epoch in pbar:
        # Training step
        optimizer.zero_grad()

        attn_scores = torch.matmul(X_train_split, attn_vector)
        # print(attn_scores)
        # Replace exactly zeros with -10000
        with torch.no_grad():
            attn_scores[attn_scores == 0] = -10000
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = einops.einsum(
            attn_weights, X_train_split, "batch seq_len, batch seq_len d -> batch d"
        )
        logits = torch.matmul(weighted_sum, output_vector) + bias

        # Forward pass on training data
        # logits = torch.matmul(last_vectors, output_vector) + bias  # Shape: [batch_size]

        # Calculate loss with L2 regularization
        l2_reg = l2_lambda * (
            torch.norm(output_vector) ** 2 + torch.norm(attn_vector) ** 2
        )
        loss = criterion(logits, y_train_split.float()) + l2_reg

        # Backward pass
        loss.backward()

        # Calculate train AUC
        train_auc = roc_auc_score(
            y_train_split.detach().cpu().numpy(), logits.detach().cpu().numpy()
        )

        # Calculate validation AUC
        with torch.no_grad():
            val_attn_scores = torch.matmul(val_vectors, attn_vector)
            val_attn_scores[val_attn_scores == 0] = -10000
            val_attn_weights = torch.softmax(val_attn_scores, dim=1)
            val_weighted_sum = einops.einsum(
                val_attn_weights,
                val_vectors,
                "batch seq_len, batch seq_len d -> batch d",
            )
            val_logits = torch.matmul(val_weighted_sum, output_vector) + bias
            val_auc = roc_auc_score(
                y_train[val_idx].cpu().numpy(), val_logits.cpu().numpy()
            )

            # Save best parameters
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_output_vector = output_vector.clone().detach()
                best_attn_vector = attn_vector.clone().detach()
                best_bias = bias.clone().detach()

        optimizer.step()

        # Update progress bar with train and validation AUC
        pbar.set_description(
            f"Train AUC: {train_auc:.3f}, "
            f"Val AUC: {val_auc:.3f}, "
            f"L: {loss.item():.3f}, "
            f"B: {bias.item():.3f}, "
            f"A_N: {torch.norm(attn_vector).item():.3f}, "
            f"O_N: {torch.norm(output_vector).item():.3f}, "
            f"A_NZ: {(attn_weights > 0).float().mean().item():.3f}"
        )

    # Calculate test AUC using best parameters
    with torch.no_grad():
        test_attn_scores = torch.matmul(X_test.cpu(), best_attn_vector.cpu())
        test_attn_scores[test_attn_scores == 0] = -10000
        test_attn_weights = torch.softmax(test_attn_scores, dim=1)
        test_weighted_sum = einops.einsum(
            test_attn_weights, X_test.cpu(), "batch seq_len, batch seq_len d -> batch d"
        )
        test_logits = (
            torch.matmul(test_weighted_sum, best_output_vector.cpu()) + best_bias.cpu()
        )
        test_auc = roc_auc_score(y_test.cpu().numpy(), test_logits.numpy())

    metrics = {
        "train_auc": train_auc,
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "best_output_vector": best_output_vector,
        "best_attn_vector": best_attn_vector,
        "best_bias": best_bias,
    }
    return metrics


def train_attn_probing_on_model_acts(dataset, layer):
    save_path = (
        f"data/consolidated_probing_{model_name}/{dataset}_{layer}_attn_probing.pkl"
    )
    if os.path.exists(save_path):
        return

    model_acts = load_model_acts(dataset)

    size = dataset_sizes[dataset]
    num_train = min(size - 100, 1024)
    num_test = size - num_train
    y = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio=0.5, seed=42
    )

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train_model = model_acts[train_indices]
    X_test_model = model_acts[test_indices]

    metrics = train_attn_probing(X_train_model, X_test_model, y_train, y_test)
    metrics["dataset"] = dataset
    metrics["layer"] = layer

    with open(save_path, "wb") as f:
        pkl.dump(metrics, f)


# %%

if "attn_probing" in to_run_list:
    for dataset in tqdm(datasets):
        try:
            train_attn_probing_on_model_acts(dataset, layer, sae_id)
        except FileNotFoundError:
            continue
