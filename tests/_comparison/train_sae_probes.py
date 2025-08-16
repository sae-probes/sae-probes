import argparse
import os
import pickle as pkl
import random
import warnings

import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from .utils_data import (
    corrupt_ytrain,
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_training_sizes,
)
from .utils_sae import get_sae_layers, get_sae_layers_extra, layer_to_sae_ids
from .utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# Constants and datasets
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()
train_sizes = get_training_sizes()
corrupt_fracs = get_corrupt_frac()
fracs = get_class_imbalance()


def load_activations(path):
    return torch.load(path, weights_only=True).to_dense().float()


# Normal setting functions
def get_sae_paths(
    dataset,
    layer,
    sae_id,
    reg_type,
    binarize=False,
    model_name="gemma-2-9b",
    setting="normal",
    extra_string="_",
):
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

    if extra_string != "_":
        if not extra_string.endswith("_"):
            extra_string = extra_string + "_"
        if not extra_string.startswith("_"):
            extra_string = "_" + extra_string

    extra_save_string = extra_string
    extra_load_string = extra_string
    save_setting = setting
    load_setting_train = setting
    load_setting_test = setting
    if setting == "label_noise":
        extra_load_string = "_"
        load_setting_train = "normal"
        load_setting_test = "normal"

    if setting == "OOD":
        load_setting_train = "normal"

    save_path = f"data/sae_probes_{model_name}/{save_setting}_setting/{description_string}{extra_save_string}{reg_type}.pkl"
    train_path = f"data/sae_activations_{model_name}/{load_setting_train}_setting/{description_string}{extra_load_string}X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/{load_setting_test}_setting/{description_string}{extra_load_string}X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/{load_setting_train}_setting/{description_string}{extra_load_string}y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/{load_setting_test}_setting/{description_string}{extra_load_string}y_test.pt"
    return {
        "save_path": save_path,
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }


def get_sorted_indices(X_train_sae, y_train):
    X_train_diff = X_train_sae[y_train == 1].mean(dim=0) - X_train_sae[
        y_train == 0
    ].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    return sorted_indices


def get_sorted_indices_new(X_train_sae, y_train):
    col_sums = X_train_sae.sum(dim=0)
    col_nonzero_counts = (X_train_sae != 0).sum(dim=0)
    col_means = col_sums / (col_nonzero_counts + 1e-6)

    # Divide each col by the average of the col
    X_train_sae_normalized = X_train_sae / (col_means + 1e-6)
    X_train_diff = X_train_sae_normalized[y_train == 1].mean(
        dim=0
    ) - X_train_sae_normalized[y_train == 0].mean(dim=0)
    sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
    return sorted_indices


def get_sae_paths_wrapper(
    dataset,
    layer,
    sae_id,
    reg_type,
    setting,
    model_name="gemma-2-9b",
    binarize=False,
    num_train=None,
    corrupt_frac=None,
    frac=None,
):
    if setting == "normal":
        extra_string = "_"
        assert num_train is None
        assert corrupt_frac is None
        assert frac is None
    elif setting == "scarcity":
        extra_string = f"{num_train}"
        assert num_train is not None
        assert corrupt_frac is None
        assert frac is None
    elif setting == "label_noise":
        extra_string = f"{corrupt_frac}"
        assert corrupt_frac is not None
        assert num_train is None
        assert frac is None
    elif setting == "class_imbalance":
        extra_string = f"frac{frac}"
        assert frac is not None
        assert num_train is None
        assert corrupt_frac is None
    elif setting == "OOD":
        extra_string = "_"
        assert num_train is None
        assert corrupt_frac is None
        assert frac is None
    else:
        raise ValueError(f"Invalid setting: {setting}")

    return get_sae_paths(
        dataset,
        layer,
        sae_id,
        reg_type,
        binarize,
        model_name,
        setting,
        extra_string=extra_string,
    )


def run_baseline(
    dataset,
    layer,
    sae_id,
    reg_type,
    setting,
    model_name="gemma-2-9b",
    binarize=False,
    num_train=None,
    corrupt_frac=None,
    frac=None,
):
    # Get appropriate paths based on setting
    paths = get_sae_paths_wrapper(
        dataset,
        layer,
        sae_id,
        reg_type,
        setting,
        model_name,
        binarize,
        num_train,
        corrupt_frac,
        frac,
    )

    train_path, test_path, y_train_path, y_test_path = (
        paths["train_path"],
        paths["test_path"],
        paths["y_train_path"],
        paths["y_test_path"],
    )

    # Check if all required files exist
    if not all(
        os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]
    ):
        print(
            f"Missing activation files for dataset {dataset}, layer {layer}, SAE {sae_id}"
        )
        print(train_path, test_path, y_train_path, y_test_path)
        return False

    X_train_sae = load_activations(train_path)
    X_test_sae = load_activations(test_path)
    y_train = load_activations(y_train_path)
    y_test = load_activations(y_test_path)

    # Handle special cases for different settings
    if setting == "label_noise":
        y_train = corrupt_ytrain(y_train.numpy(), corrupt_frac)

    # Set k values based on setting
    if setting == "normal":
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    else:
        ks = [16, 128]

    all_metrics = []
    sorted_indices = get_sorted_indices_new(X_train_sae, y_train)

    for k in tqdm(ks):
        top_by_average_diff = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, top_by_average_diff]
        X_test_filtered = X_test_sae[:, top_by_average_diff]

        if binarize and setting == "normal":
            X_train_filtered = X_train_filtered > 1
            X_test_filtered = X_test_filtered > 1

        if reg_type == "l1":
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l1",
            )
        else:
            metrics = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                plot=False,
                n_jobs=-1,
                parallel=False,
                penalty="l2",
            )

        # Add metadata to metrics
        metrics.update(
            {
                "k": k,
                "dataset": dataset,
                "layer": layer,
                "sae_id": sae_id,
                "reg_type": reg_type,
                "binarize": binarize,
            }
        )

        # Add setting-specific metadata
        if setting == "scarcity":
            metrics["num_train"] = num_train
        elif setting == "label_noise":
            metrics["corrupt_frac"] = corrupt_frac
        elif setting == "class_imbalance":
            metrics["frac"] = frac

        all_metrics.append(metrics)

    print(f"Saving results to {paths['save_path']}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pkl.dump(all_metrics, f)

    return True


def run_baselines(
    reg_type,
    model_name,
    setting,
    binarize=False,
    target_sae_id=None,
    randomize_order=False,
):
    layers = get_sae_layers(model_name)
    if model_name == "gemma-2-9b" and setting == "normal":
        layers = get_sae_layers_extra(model_name)
    while True:
        found_missing = False
        if setting == "OOD":
            datasets = get_OOD_datasets()
        if randomize_order:
            loop_datasets = random.sample(datasets, len(datasets))
            loop_layers = random.sample(layers, len(layers))
        else:
            loop_datasets = datasets
            loop_layers = layers

        for dataset in loop_datasets:
            loop_layers = random.sample(layers, len(layers))
            for layer in loop_layers:
                if target_sae_id is not None:
                    sae_ids = [target_sae_id]
                else:
                    sae_ids = layer_to_sae_ids(layer, model_name)
                    if model_name == "gemma-2-9b" and setting != "normal":
                        sae_ids = [
                            "layer_20/width_16k/average_l0_408",
                            "layer_20/width_131k/average_l0_276",
                            "layer_20/width_1m/average_l0_193",
                        ]

                if randomize_order:
                    loop_sae_ids = random.sample(sae_ids, len(sae_ids))
                else:
                    loop_sae_ids = sae_ids
                for sae_id in loop_sae_ids:
                    # Handle different settings
                    if setting == "normal":
                        paths = get_sae_paths_wrapper(
                            dataset,
                            layer,
                            sae_id,
                            reg_type,
                            setting,
                            model_name,
                            binarize,
                        )
                        if not os.path.exists(paths["save_path"]):
                            found_missing = True
                            print(
                                f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}, setting {setting}"
                            )
                            success = run_baseline(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                setting,
                                model_name,
                                binarize,
                            )
                            assert success
                    elif setting == "scarcity":
                        for num_train in train_sizes:
                            if num_train > dataset_sizes[dataset] - 100:
                                continue
                            paths = get_sae_paths_wrapper(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                setting,
                                model_name,
                                binarize,
                                num_train=num_train,
                            )
                            if not os.path.exists(paths["save_path"]):
                                found_missing = True
                                print(
                                    f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, "
                                    f"reg_type {reg_type}, num_train {num_train}, setting {setting}"
                                )
                                success = run_baseline(
                                    dataset,
                                    layer,
                                    sae_id,
                                    reg_type,
                                    setting,
                                    model_name,
                                    num_train=num_train,
                                )
                                assert success
                    elif setting == "label_noise":
                        for corrupt_frac in corrupt_fracs:
                            paths = get_sae_paths_wrapper(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                setting,
                                model_name,
                                binarize,
                                corrupt_frac=corrupt_frac,
                            )
                            if not os.path.exists(paths["save_path"]):
                                found_missing = True
                                print(
                                    f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, "
                                    f"reg_type {reg_type}, corrupt_frac {corrupt_frac}, setting {setting}"
                                )
                                success = run_baseline(
                                    dataset,
                                    layer,
                                    sae_id,
                                    reg_type,
                                    setting,
                                    model_name,
                                    corrupt_frac=corrupt_frac,
                                )
                                assert success
                    elif setting == "class_imbalance":
                        for frac in fracs:
                            paths = get_sae_paths_wrapper(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                setting,
                                model_name,
                                binarize,
                                frac=frac,
                            )
                            if not os.path.exists(paths["save_path"]):
                                print(
                                    f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, "
                                    f"reg_type {reg_type}, frac {frac}, setting {setting}"
                                )
                                success = run_baseline(
                                    dataset,
                                    layer,
                                    sae_id,
                                    reg_type,
                                    setting,
                                    model_name,
                                    frac=frac,
                                )
                                assert success
                    elif setting == "OOD":
                        paths = get_sae_paths_wrapper(
                            dataset,
                            layer,
                            sae_id,
                            reg_type,
                            setting,
                            model_name,
                            binarize,
                        )
                        if not os.path.exists(paths["save_path"]):
                            found_missing = True
                            print(
                                f"Running probe for dataset {dataset}, layer {layer}, SAE {sae_id}, reg_type {reg_type}, setting {setting}"
                            )
                            success = run_baseline(
                                dataset,
                                layer,
                                sae_id,
                                reg_type,
                                setting,
                                model_name,
                                binarize,
                            )
                            assert success
                    else:
                        raise ValueError(f"Invalid setting: {setting}")

        if not found_missing:
            print(f"All {setting} probes run. Exiting.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE probes in various settings")
    parser.add_argument(
        "--reg_type",
        type=str,
        required=True,
        choices=["l1", "l2"],
        help="Regularization type",
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["normal", "scarcity", "label_noise", "class_imbalance", "OOD"],
        help="Probe training setting (normal, scarcity, label_noise, or imbalance)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"],
        help="Model name",
    )
    parser.add_argument(
        "--binarize", action="store_true", help="Whether to binarize activations"
    )
    parser.add_argument(
        "--target_sae_id", type=str, help="Target specific SAE ID (optional)"
    )
    parser.add_argument(
        "--randomize_order",
        action="store_true",
        help="Randomize order of datasets and layers",
    )

    args = parser.parse_args()

    run_baselines(
        args.reg_type,
        args.model_name,
        args.setting,
        args.binarize,
        args.target_sae_id,
        args.randomize_order,
    )
