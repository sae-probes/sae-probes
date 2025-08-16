# %%
import argparse
import os
import random
import warnings

import torch
from sklearn.exceptions import ConvergenceWarning

from .utils_data import (
    get_class_imbalance,
    get_classimabalance_num_train,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from .utils_sae import get_sae_layers, layer_to_sae_ids, sae_id_to_sae

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# %%

# Common variables
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()


# %%
# Helper functions for all settings
def save_activations(path, activation):
    """Save activations in sparse format to save space"""
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)


def load_activations(path):
    """Load activations from sparse format"""
    return torch.load(path, weights_only=True).to_dense().float()


# %%
# Normal setting functions
def get_sae_paths_normal(
    dataset, layer, sae_id, reg_type, binarize=False, model_name="gemma-2-9b"
):
    """Get paths for normal setting"""
    os.makedirs(f"data/sae_probes_{model_name}/normal_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/normal_setting", exist_ok=True)

    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if binarize:
        reg_type += "_binarized"

    save_path = f"data/sae_probes_{model_name}/normal_setting/{description_string}_{reg_type}.pkl"
    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def save_with_sae_normal(layer, sae, sae_id, model_name, device, reg_type, binarize):
    """Generate and save SAE activations for normal setting"""
    for dataset in datasets:
        paths = get_sae_paths_normal(
            dataset, layer, sae_id, reg_type, binarize, model_name
        )
        train_path, test_path, y_train_path, y_test_path = (
            paths["train_path"],
            paths["test_path"],
            paths["y_train_path"],
            paths["y_test_path"],
        )

        all_paths_exist = all(
            [
                os.path.exists(train_path),
                os.path.exists(test_path),
                os.path.exists(y_train_path),
                os.path.exists(y_test_path),
            ]
        )
        if all_paths_exist:
            continue

        size = dataset_sizes[dataset]
        num_train = min(size - 100, 1024)
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train, dataset, layer, model_name=model_name
        )

        batch_size = 128
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

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))


# %%
# Data scarcity setting functions
def get_sae_paths_scarcity(
    dataset, layer, sae_id, reg_type, num_train, model_name="gemma-2-9b"
):
    """Get paths for data scarcity setting"""
    os.makedirs(f"data/sae_probes_{model_name}/scarcity_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/scarcity_setting", exist_ok=True)

    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    save_path = f"data/sae_probes_{model_name}/scarcity_setting/{description_string}_{reg_type}_{num_train}.pkl"
    train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_test.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def save_with_sae_scarcity(layer, sae, sae_id, model_name, device, reg_type):
    """Generate and save SAE activations for data scarcity setting"""
    train_sizes = get_training_sizes()

    for dataset in datasets:
        for num_train in train_sizes:
            if num_train > dataset_sizes[dataset] - 100:
                continue

            paths = get_sae_paths_scarcity(
                dataset, layer, sae_id, reg_type, num_train, model_name
            )
            train_path, test_path, y_train_path, y_test_path = (
                paths["train_path"],
                paths["test_path"],
                paths["y_train_path"],
                paths["y_test_path"],
            )

            if all(
                os.path.exists(p)
                for p in [train_path, test_path, y_train_path, y_test_path]
            ):
                continue

            X_train, y_train, X_test, y_test = get_xy_traintest(
                num_train, dataset, layer, model_name=model_name
            )

            batch_size = 128
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

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))


# %%
# Class imbalance setting functions
def get_sae_paths_imbalance(
    dataset, layer, sae_id, reg_type, frac, model_name="gemma-2-9b"
):
    """Get paths for class imbalance setting"""
    os.makedirs(f"data/sae_probes_{model_name}/class_imbalance", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/class_imbalance", exist_ok=True)

    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    save_path = f"data/sae_probes_{model_name}/class_imbalance/{description_string}_{reg_type}_frac{frac}.pkl"
    train_path = f"data/sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/class_imbalance/{description_string}_frac{frac}_y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def save_with_sae_imbalance(layer, sae, sae_id, model_name, device, reg_type):
    """Generate and save SAE activations for class imbalance setting"""
    fracs = get_class_imbalance()

    for dataset in datasets:
        for frac in fracs:
            paths = get_sae_paths_imbalance(
                dataset, layer, sae_id, reg_type, frac, model_name
            )
            train_path, test_path, y_train_path, y_test_path = (
                paths["train_path"],
                paths["test_path"],
                paths["y_train_path"],
                paths["y_test_path"],
            )

            if os.path.exists(train_path):
                continue

            num_train, num_test = get_classimabalance_num_train(dataset)
            X_train, y_train, X_test, y_test = get_xy_traintest_specify(
                num_train,
                dataset,
                layer,
                pos_ratio=frac,
                model_name=model_name,
                num_test=num_test,
            )

            batch_size = 128
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

            save_activations(train_path, X_train_sae)
            save_activations(test_path, X_test_sae)
            save_activations(y_train_path, torch.tensor(y_train))
            save_activations(y_test_path, torch.tensor(y_test))


# %%
# Process SAEs for a specific model and setting
def process_model_setting(
    model_name, setting, device, reg_type, binarize, randomize_order
):
    print(f"Running SAE activation generation for {model_name} in {setting} setting")

    layers = get_sae_layers(model_name)
    found_missing = False

    for layer in layers:
        sae_ids = layer_to_sae_ids(layer, model_name)

        if randomize_order:
            random.shuffle(sae_ids)

        for sae_id in sae_ids:
            print(f"Processing SAE: {sae_id}")

            # Check if we need to generate activations for this SAE
            missing_data = False

            if setting == "normal":
                # Check normal setting
                for dataset in datasets:
                    paths = get_sae_paths_normal(
                        dataset, layer, sae_id, reg_type, binarize, model_name
                    )
                    if not all(
                        os.path.exists(p)
                        for p in [
                            paths["train_path"],
                            paths["test_path"],
                            paths["y_train_path"],
                            paths["y_test_path"],
                        ]
                    ):
                        print(f"Missing data for dataset {dataset}")
                        missing_data = True
                        break

                if missing_data:
                    try:
                        sae = sae_id_to_sae(sae_id, model_name, device)
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_normal(
                            layer, sae, sae_id, model_name, device, reg_type, binarize
                        )
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue

            elif setting == "scarcity":
                # Check data scarcity setting
                train_sizes = get_training_sizes()
                for dataset in datasets:
                    for num_train in train_sizes:
                        if num_train > dataset_sizes[dataset] - 100:
                            continue
                        paths = get_sae_paths_scarcity(
                            dataset, layer, sae_id, reg_type, num_train, model_name
                        )
                        if not all(
                            os.path.exists(p)
                            for p in [
                                paths["train_path"],
                                paths["test_path"],
                                paths["y_train_path"],
                                paths["y_test_path"],
                            ]
                        ):
                            print(
                                f"Missing data for dataset {dataset}, num_train {num_train}"
                            )
                            missing_data = True
                            break
                    if missing_data:
                        break

                if missing_data:
                    try:
                        sae = sae_id_to_sae(sae_id, model_name, device)
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_scarcity(
                            layer, sae, sae_id, model_name, device, reg_type
                        )
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue

            elif setting == "imbalance":
                # Check class imbalance setting
                fracs = get_class_imbalance()
                for dataset in datasets:
                    for frac in fracs:
                        paths = get_sae_paths_imbalance(
                            dataset, layer, sae_id, reg_type, frac, model_name
                        )
                        if not all(
                            os.path.exists(p)
                            for p in [
                                paths["train_path"],
                                paths["test_path"],
                                paths["y_train_path"],
                                paths["y_test_path"],
                            ]
                        ):
                            print(f"Missing data for dataset {dataset}, frac {frac}")
                            missing_data = True
                            break
                    if missing_data:
                        break

                if missing_data:
                    try:
                        sae = sae_id_to_sae(sae_id, model_name, device)
                        print(f"Generating SAE data for layer {layer}, SAE {sae_id}")
                        save_with_sae_imbalance(
                            layer, sae, sae_id, model_name, device, reg_type
                        )
                        found_missing = True
                        break
                    except Exception as e:
                        print(f"Error loading SAE {sae_id}: {e}")
                        continue

        if found_missing:
            break

    if not found_missing:
        print(
            f"All SAE activations for {model_name} in {setting} setting have been generated!"
        )

    return found_missing


# %%
# Main function to process all models and settings
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"],
    )
    parser.add_argument(
        "--setting", type=str, default=None, choices=["normal", "scarcity", "imbalance"]
    )
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--reg_type", type=str, choices=["l1", "l2"], default="l1")
    parser.add_argument(
        "--randomize_order",
        action="store_true",
        help="Randomize the order of datasets and settings, useful for parallelizing",
    )

    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    setting = args.setting
    binarize = args.binarize
    reg_type = args.reg_type
    randomize_order = args.randomize_order

    model_names = ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]
    settings = ["normal", "scarcity", "imbalance", "noise", "consolidated"]

    # If specific model and setting are provided via command line, use those
    if model_name is not None and setting is not None:
        process_model_setting(model_name, setting, device, reg_type, binarize)
        exit(0)

    # Otherwise, loop through all models and settings
    for curr_model_name in model_names:
        if randomize_order:
            random.shuffle(settings)
        for curr_setting in settings:
            print(f"\n{'=' * 50}")
            print(f"Processing {curr_model_name} in {curr_setting} setting")
            print(f"{'=' * 50}\n")
            do_loop = True
            while do_loop:
                do_loop = process_model_setting(
                    curr_model_name,
                    curr_setting,
                    device,
                    reg_type,
                    binarize,
                    randomize_order,
                )
