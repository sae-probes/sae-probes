import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Literal

import torch
from sae_lens import SAE
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from sae_probes.constants import (
    DEFAULT_RESULTS_PATH,
    RegType,
    Setting,
)
from sae_probes.generate_model_activations import ensure_dataset_activations
from sae_probes.generate_sae_activations import generate_sae_activations
from sae_probes.latent_selection import (
    SelectionFn,
    SelectionMethod,
    mean_diff_selection_fn,
    selection_method_name,
)
from sae_probes.latent_selection import get_sorted_indices as get_sorted_indices
from sae_probes.latent_selection import (
    mean_act_normalization as mean_act_normalization,
)
from sae_probes.utils_data import (
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_training_sizes,
)
from sae_probes.utils_tmp import resolve_model_cache_path
from sae_probes.utils_training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)


# Constants and datasets
DATASET_SIZES = get_dataset_sizes()
DATASETS = get_numbered_binary_tags()
TRAIN_SIZES = get_training_sizes()
CORRUPT_FRACS = get_corrupt_frac()
FRACS = get_class_imbalance()


def load_activations(path):
    return torch.load(path, weights_only=True).to_dense().float()


# Normal setting functions
def get_save_metrics_path(
    dataset: str,
    hook_name: str,
    reg_type: RegType,
    model_name: str,
    sae_results_path: str | Path,
    binarize: bool = False,
    setting: Setting = "normal",
    num_train: int | None = None,
    corrupt_frac: float | None = None,
    frac: float | None = None,
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
):
    description_string = f"{dataset}_{hook_name}"

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
    elif setting == "imbalance":
        extra_string = f"frac{frac}"
        assert frac is not None
        assert num_train is None
        assert corrupt_frac is None

    reg_type_str: str = reg_type
    if binarize:
        reg_type_str = f"{reg_type_str}_binarized"
    # the default method keeps the original file names, so existing results stay valid
    if selection_method != "mean_diff":
        reg_type_str = f"{reg_type_str}_sel-{selection_method_name(selection_method)}"

    if extra_string != "_":
        if not extra_string.endswith("_"):
            extra_string = extra_string + "_"
        if not extra_string.startswith("_"):
            extra_string = "_" + extra_string

    extra_save_string = extra_string
    save_setting = setting

    save_path = (
        Path(sae_results_path)
        / f"sae_probes_{model_name}/{save_setting}_setting/{description_string}{extra_save_string}{reg_type_str}.json"
    )
    return save_path


def resolve_selection_method(
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
    mean_diff_normalization: (
        Literal["mean", "none"] | Callable[[torch.Tensor], torch.Tensor] | None
    ) = None,
) -> SelectionMethod | SelectionFn:
    """Map the deprecated `mean_diff_normalization` argument onto `selection_method`."""
    if mean_diff_normalization is None:
        return selection_method
    warnings.warn(
        "`mean_diff_normalization` is deprecated and will be removed in a future "
        'version. Use `selection_method` instead: "mean_diff" replaces "mean", '
        '"raw_mean_diff" replaces "none", and a custom normalization can be passed as '
        "`selection_method=mean_diff_selection_fn(normalize_fn)`.",
        DeprecationWarning,
        stacklevel=3,
    )
    if selection_method != "mean_diff":
        raise ValueError(
            "Pass `selection_method` or the deprecated `mean_diff_normalization`, not both"
        )
    if mean_diff_normalization == "mean":
        return "mean_diff"
    if mean_diff_normalization == "none":
        return "raw_mean_diff"
    if callable(mean_diff_normalization):
        return mean_diff_selection_fn(mean_diff_normalization)
    raise ValueError(f"Invalid mean_diff_normalization: {mean_diff_normalization}")


def run_sae_eval(
    sae: SAE,
    dataset: str,
    hook_name: str,
    reg_type: RegType,
    setting: Setting,
    model_name: str,
    model_cache_path: str | Path,
    binarize: bool = False,
    num_train: int | None = None,
    corrupt_frac: float | None = None,
    frac: float | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    ks: list[int] | None = None,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    mean_diff_normalization: (
        Literal["mean", "none"] | Callable[[torch.Tensor], torch.Tensor] | None
    ) = None,
    seed: int = 42,
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
):
    """
    Train k-sparse probes on the k SAE latents ranked best by `selection_method` on
    the training set: "mean_diff" (default, the method from the original paper),
    "raw_mean_diff", "correlation", "firing_rate", "auc", "accuracy", "forward", or a
    callable taking (X_train, y_train) and returning one score per latent (larger is
    better).
    See `sae_probes.latent_selection` for what each method does.

    With `binarize=True` (normal setting), latents are ranked on the binarized
    activations that the probe is trained on. Versions up to 0.4.2 ranked latents on
    the continuous activations and binarized afterwards.

    `mean_diff_normalization` is deprecated; use `selection_method`.
    """
    selection_method = resolve_selection_method(
        selection_method, mean_diff_normalization
    )
    activations = generate_sae_activations(
        sae=sae,
        setting=setting,
        dataset=dataset,
        hook_name=hook_name,
        model_name=model_name,
        device=device,
        num_train=num_train,
        frac=frac,
        batch_size=batch_size,
        model_cache_path=model_cache_path,
        seed=seed,
    )

    X_train_sae = activations.X_train
    X_test_sae = activations.X_test
    y_train = activations.y_train
    y_test = activations.y_test

    # Set k values based on setting
    if ks is None:
        if setting == "normal":
            ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            ks = [16, 128]

    all_metrics = []
    binarize_probe_inputs = binarize and setting == "normal"
    # rank latents on what the probe will see
    X_train_for_selection = X_train_sae != 0 if binarize_probe_inputs else X_train_sae
    sorted_indices = get_sorted_indices(
        X_train_for_selection, y_train, selection_method, k_max=max(ks)
    )

    for k in tqdm(ks):
        selected_indices = sorted_indices[:k]
        X_train_filtered = X_train_sae[:, selected_indices]
        X_test_filtered = X_test_sae[:, selected_indices]

        if binarize_probe_inputs:
            X_train_filtered = X_train_filtered != 0
            X_test_filtered = X_test_filtered != 0

        results = find_best_reg(
            X_train=X_train_filtered,
            y_train=y_train,
            X_test=X_test_filtered,
            y_test=y_test,
            n_jobs=-1,
            parallel=False,
            penalty=reg_type,
            seed=seed,
        )
        metrics = asdict(results.metrics)

        # Add metadata to metrics
        metrics.update(
            {
                "k": k,
                "dataset": dataset,
                "hook_name": hook_name,
                "reg_type": reg_type,
                "binarize": binarize,
                "selection_method": selection_method_name(selection_method),
                "indices": selected_indices.tolist(),
            }
        )

        # Add setting-specific metadata
        if setting == "scarcity":
            metrics["num_train"] = num_train
        elif setting == "imbalance":
            metrics["frac"] = frac

        all_metrics.append(metrics)

    save_path = get_save_metrics_path(
        dataset=dataset,
        hook_name=hook_name,
        reg_type=reg_type,
        binarize=binarize,
        model_name=model_name,
        setting=setting,
        num_train=num_train,
        corrupt_frac=corrupt_frac,
        frac=frac,
        sae_results_path=results_path,
        selection_method=selection_method,
    )

    print(f"Saving results to {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    return True


def run_sae_evals(
    sae: SAE,
    model_name: str,
    hook_name: str,
    reg_type: RegType,
    setting: Setting,
    ks: list[int] | None = None,
    binarize: bool = False,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    model_cache_path: str | Path | None = None,
    datasets: list[str] | None = None,
    device: str = "cuda",
    mean_diff_normalization: (
        Literal["mean", "none"] | Callable[[torch.Tensor], torch.Tensor] | None
    ) = None,
    seed: int = 42,
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
):
    """
    Run the sparse probing benchmark for an SAE on every dataset.

    `selection_method` chooses how the k latents of each k-sparse probe are picked on
    the training set: "mean_diff" (default, the method from the original paper),
    "raw_mean_diff", "correlation", "firing_rate", "auc", "accuracy", "forward", or a
    callable taking (X_train, y_train) and returning one score per latent. See
    `sae_probes.latent_selection` for what each method does. Results for non-default
    methods are saved under separate file names.

    `mean_diff_normalization` is deprecated; use `selection_method`.
    """
    selection_method = resolve_selection_method(
        selection_method, mean_diff_normalization
    )
    if datasets is None:
        datasets = DATASETS
    with resolve_model_cache_path(model_cache_path) as resolved_cache_path:
        ensure_dataset_activations(
            model_name=model_name,
            dataset_short_names=datasets,
            hook_names=[hook_name],
            model_cache_path=resolved_cache_path,
            device=device,
        )
        for dataset in datasets:
            # Handle different settings
            if setting == "normal":
                save_path = get_save_metrics_path(
                    dataset=dataset,
                    hook_name=hook_name,
                    reg_type=reg_type,
                    binarize=binarize,
                    model_name=model_name,
                    setting=setting,
                    sae_results_path=results_path,
                    selection_method=selection_method,
                )
                if save_path.exists():
                    print(
                        f"Skipping dataset {dataset}, hook {hook_name}, reg_type {reg_type}, setting {setting}"
                    )
                else:
                    print(
                        f"Running probe for dataset {dataset}, hook {hook_name}, reg_type {reg_type}, setting {setting}"
                    )
                    success = run_sae_eval(
                        sae=sae,
                        dataset=dataset,
                        hook_name=hook_name,
                        reg_type=reg_type,
                        setting=setting,
                        model_name=model_name,
                        model_cache_path=resolved_cache_path,
                        binarize=binarize,
                        ks=ks,
                        results_path=results_path,
                        device=device,
                        selection_method=selection_method,
                        seed=seed,
                    )
                    assert success
            elif setting == "scarcity":
                for num_train in TRAIN_SIZES:
                    if num_train > DATASET_SIZES[dataset] - 100:
                        continue
                    save_path = get_save_metrics_path(
                        dataset=dataset,
                        hook_name=hook_name,
                        reg_type=reg_type,
                        binarize=binarize,
                        model_name=model_name,
                        setting=setting,
                        num_train=num_train,
                        sae_results_path=results_path,
                        selection_method=selection_method,
                    )
                    if save_path.exists():
                        print(
                            f"Skipping dataset {dataset}, hook {hook_name}, reg_type {reg_type}, setting {setting}, num_train {num_train}"
                        )
                    else:
                        print(
                            f"Running probe for dataset {dataset}, hook {hook_name}, reg_type {reg_type}, num_train {num_train}, setting {setting}"
                        )
                        success = run_sae_eval(
                            sae=sae,
                            dataset=dataset,
                            hook_name=hook_name,
                            reg_type=reg_type,
                            setting=setting,
                            model_name=model_name,
                            model_cache_path=resolved_cache_path,
                            num_train=num_train,
                            ks=ks,
                            results_path=results_path,
                            device=device,
                            selection_method=selection_method,
                            seed=seed,
                        )
                        assert success
            elif setting == "imbalance":
                for frac in FRACS:
                    save_path = get_save_metrics_path(
                        dataset=dataset,
                        hook_name=hook_name,
                        reg_type=reg_type,
                        binarize=binarize,
                        model_name=model_name,
                        setting=setting,
                        frac=frac,
                        sae_results_path=results_path,
                        selection_method=selection_method,
                    )
                    if save_path.exists():
                        print(
                            f"Skipping dataset {dataset}, hook {hook_name}, reg_type {reg_type}, frac {frac}, setting {setting}"
                        )
                    else:
                        print(
                            f"Running probe for dataset {dataset}, hook {hook_name}, reg_type {reg_type}, frac {frac}, setting {setting}"
                        )
                        success = run_sae_eval(
                            sae=sae,
                            dataset=dataset,
                            hook_name=hook_name,
                            reg_type=reg_type,
                            setting=setting,
                            model_name=model_name,
                            model_cache_path=resolved_cache_path,
                            frac=frac,
                            ks=ks,
                            results_path=results_path,
                            device=device,
                            selection_method=selection_method,
                            seed=seed,
                        )
                        assert success
            else:
                raise ValueError(f"Invalid setting: {setting}")
