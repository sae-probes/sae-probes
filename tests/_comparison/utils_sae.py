# %%
import numpy as np
import pandas as pd
import torch
from handle_sae_bench_saes import get_gemma_2_2b_sae_ids, load_gemma_2_2b_sae
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from .utils_data import get_xy_glue, get_xy_OOD, get_xyvals


def get_gemma_2_9b_sae_ids(layer):
    all_gemma_scope_saes = get_pretrained_saes_directory()[
        "gemma-scope-9b-pt-res"
    ].saes_map
    all_sae_ids = [
        sae_id
        for sae_id in all_gemma_scope_saes
        if sae_id.split("/")[0] == f"layer_{layer}"
    ]
    return all_sae_ids


def get_gemma_2_9b_sae_ids_largest_l0s(layer, width_restriction=["16k", "131k", "1m"]):
    all_sae_ids = get_gemma_2_9b_sae_ids(layer)
    width_to_largest_sae_id = {}
    width_to_largest_l0 = {}
    for sae_id in all_sae_ids:
        width = sae_id.split("/")[1].split("_")[-1]
        l0 = sae_id.split("/")[2].split("_")[-1]
        if width not in width_restriction:
            continue
        if width not in width_to_largest_sae_id:
            width_to_largest_sae_id[width] = sae_id
            width_to_largest_l0[width] = l0
        elif int(l0) > int(width_to_largest_l0[width]):
            width_to_largest_sae_id[width] = sae_id
            width_to_largest_l0[width] = l0
    return list(width_to_largest_sae_id.values())


def load_gemma_2_9b_sae(sae_id):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res",
        sae_id=sae_id,
        device="cpu",
    )
    return sae


def load_llama_3_1_8b_sae(sae_id):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="llama_scope_lxr_32x",
        sae_id=sae_id,
        device="cpu",
    )
    return sae


def layer_to_sae_ids(layer, model_name):
    if model_name == "gemma-2-9b":
        return get_gemma_2_9b_sae_ids(layer)
    elif model_name == "llama-3.1-8b":
        return [f"l{layer}r_32x"]
    elif model_name == "gemma-2-2b":
        assert layer == 12
        return get_gemma_2_2b_sae_ids(layer)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def sae_id_to_sae(sae_id, model_name, device):
    if model_name == "gemma-2-9b":
        return load_gemma_2_9b_sae(sae_id).to(device)
    elif model_name == "llama-3.1-8b":
        return load_llama_3_1_8b_sae(sae_id).to(device)
    elif model_name == "gemma-2-2b":
        return load_gemma_2_2b_sae(sae_id).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def get_xy_OOD_sae(
    dataset,
    k=128,
    model_name="gemma-2-9b",
    layer=20,
    return_indices=False,
    num_train=1024,
):
    _, y_test = get_xy_OOD(dataset)
    _, y_train = get_xyvals(dataset, layer=layer, model_name=model_name, MAX_AMT=1500)
    X_test = (
        torch.load(
            f"data/sae_activations_{model_name}_OOD/{dataset}_OOD.pt",
            weights_only=False,
        )
        .to_dense()
        .cpu()
    )
    X_train = (
        torch.load(f"data/sae_activations_{model_name}/{dataset}.pt", weights_only=True)
        .to_dense()
        .cpu()
    )
    # Get indices for each class
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Take 512 samples from each class
    pos_selected = pos_indices[: num_train // 2]
    neg_selected = neg_indices[: num_train // 2]

    # Combine and shuffle indices
    selected_indices = np.concatenate([pos_selected, neg_selected])
    shuffled_indices = np.random.permutation(selected_indices)

    # Update X_train and y_train with balanced samples
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    top_by_average_diff = sorted_indices[:k]
    # print(top_by_average_diff)
    X_train_filtered = X_train[:, top_by_average_diff]
    X_test_filtered = X_test[:, top_by_average_diff]
    if return_indices:
        return X_train_filtered, y_train, X_test_filtered, y_test, top_by_average_diff
    return X_train_filtered, y_train, X_test_filtered, y_test


def get_xy_glue_sae(toget="ensemble", k=128):
    dataset = "87_glue_cola"
    _, y_test = get_xy_glue(toget=toget)
    _, y_train = get_xyvals(dataset, layer=20, model_name="gemma-2-9b", MAX_AMT=1500)
    X_test = (
        torch.load(
            "data/dataset_investigate/sae_gemma-2-9b_87_glue_cola.pt",
            weights_only=False,
        )
        .to_dense()
        .cpu()
    )
    X_train = (
        torch.load(
            f"data/sae_activations_gemma-2-9b_1m/{dataset}.pt", weights_only=True
        )
        .to_dense()
        .cpu()
    )
    # Get indices for each class
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Take 512 samples from each class
    pos_selected = pos_indices[:512]
    neg_selected = neg_indices[:512]

    # Combine and shuffle indices
    selected_indices = np.concatenate([pos_selected, neg_selected])
    shuffled_indices = np.random.permutation(selected_indices)

    # Update X_train and y_train with balanced samples
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    if k == 1:
        top_by_average_diff = sorted_indices[2:3]
        print(top_by_average_diff)
    else:
        top_by_average_diff = sorted_indices[:k]
    X_train_filtered = X_train[:, top_by_average_diff]
    X_test_filtered = X_test[:, top_by_average_diff]
    return X_train_filtered, y_train, X_test_filtered, y_test


def get_grammar_feature_examples():
    _, _, X, _ = get_xy_glue_sae(k=1)
    # Read prompts and get their lengths
    df = pd.read_csv("results/investigate/87_glue_cola_investigate.csv")
    prompts = df["prompt"].tolist()

    _, yog = get_xy_glue(toget="original_target")
    _, yens = get_xy_glue(toget="ensemble")

    # Get indices where original target is 1
    valid_indices = torch.tensor([i for i in range(len(yog)) if yog[i] == 1])
    X_valid = X[valid_indices]

    # Get indices of top 5 highest feature values among valid examples
    top_5_relative_idx = torch.argsort(X_valid[:, 0], descending=True)[:5]
    top_5_idx = valid_indices[top_5_relative_idx]

    # Print table
    print("\nPrompt | Original | Ensemble | Feature Fired")
    print("-" * 50)
    for idx in top_5_idx:
        print(
            f"{prompts[idx]:<30} | {yog[idx]:<8} | {yens[idx]:<8} | {X[idx].item():.2f}"
        )


# _,_,_,_, indices = get_xy_OOD_sae(dataset = '66_living-room', k = 1, model_name = 'gemma-2-9b', layer = 20, return_indices = True)
# print(indices)
# get_grammar_feature_examples()
# get_grammar_feature_examples()
# Xtrain, ytrain, Xtest, ytest = get_xy_glue_sae(k = 1)
# get_xy_OOD_sae('7_hist_fig_ispolitician', k = 8)
# print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
# X_train_filtered, y_train, X_test_filtered, y_test = get_xy_OOD_sae('87_glue_cola')
# print(f"X_train_filtered shape: {X_train_filtered.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test_filtered shape: {X_test_filtered.shape}")
# print(f"y_test shape: {y_test.shape}")


def get_sae_layers(model_name):
    if model_name == "gemma-2-9b":
        return [20]
    elif model_name == "llama-3.1-8b":
        return [16]
    elif model_name == "gemma-2-2b":
        return [12]
