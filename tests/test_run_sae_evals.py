import importlib
import json
from pathlib import Path

import pytest
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_probes.constants import RegType, Setting
from sae_probes.generate_sae_activations import generate_sae_activations
from sae_probes.latent_selection import SelectionMethod, mean_diff_selection_fn
from sae_probes.run_sae_evals import (
    get_save_metrics_path,
    get_sorted_indices,
    mean_act_normalization,
    resolve_selection_method,
    run_sae_eval,
    run_sae_evals,
)
from sae_probes.utils_data import (
    get_class_imbalance,
    get_dataset_sizes,
    get_training_sizes,
)
from sae_probes.utils_training import find_best_reg
from tests.helpers import TEST_DATASET_NAME, generate_model_activations

# sae_probes/__init__.py re-exports the run_sae_evals function, which shadows the
# submodule, so grab the module itself for monkeypatching
run_sae_evals_module = importlib.import_module("sae_probes.run_sae_evals")


def test_mean_act_normalization_basic() -> None:
    # Test basic normalization with positive values
    X_sae = torch.tensor([[1.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 4.0, 6.0]])

    result = mean_act_normalization(X_sae)

    # Check that result has same shape
    assert result.shape == X_sae.shape

    # Manually calculate expected values
    # Column 0: values [1.0, 2.0, 0.0], non-zero count = 2, sum = 3.0, mean = 1.5
    # Column 1: values [2.0, 0.0, 4.0], non-zero count = 2, sum = 6.0, mean = 3.0
    # Column 2: values [0.0, 3.0, 6.0], non-zero count = 2, sum = 9.0, mean = 4.5

    expected = torch.tensor(
        [
            [1.0 / 1.5, 2.0 / 3.0, 0.0 / 4.5],
            [2.0 / 1.5, 0.0 / 3.0, 3.0 / 4.5],
            [0.0 / 1.5, 4.0 / 3.0, 6.0 / 4.5],
        ]
    )

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_mean_act_normalization_with_negative_values() -> None:
    # Test that negative values are handled correctly (abs is taken)
    X_sae = torch.tensor([[-1.0, 2.0, 0.0], [2.0, 0.0, -3.0], [0.0, -4.0, 6.0]])

    result = mean_act_normalization(X_sae)

    # Check that result has same shape
    assert result.shape == X_sae.shape

    # Manually calculate expected values using abs for means
    # Column 0: abs values [1.0, 2.0, 0.0], non-zero count = 2, sum = 3.0, mean = 1.5
    # Column 1: abs values [2.0, 0.0, 4.0], non-zero count = 2, sum = 6.0, mean = 3.0
    # Column 2: abs values [0.0, 3.0, 6.0], non-zero count = 2, sum = 9.0, mean = 4.5

    expected = torch.tensor(
        [
            [-1.0 / 1.5, 2.0 / 3.0, 0.0 / 4.5],
            [2.0 / 1.5, 0.0 / 3.0, -3.0 / 4.5],
            [0.0 / 1.5, -4.0 / 3.0, 6.0 / 4.5],
        ]
    )

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_mean_act_normalization_all_zeros_column() -> None:
    # Test handling of columns with all zeros
    X_sae = torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

    result = mean_act_normalization(X_sae)

    # Check that result has same shape
    assert result.shape == X_sae.shape

    # Column 1 has all zeros, so col_means[1] = 0 / (0 + 1e-6) = 0
    # Division by (0 + 1e-6) should give very large values
    # Column 0: mean = 3.0/2 = 1.5
    # Column 2: mean = 6.0/2 = 3.0

    expected = torch.tensor(
        [
            [1.0 / 1.5, 0.0 / 1e-6, 2.0 / 3.0],
            [2.0 / 1.5, 0.0 / 1e-6, 0.0 / 3.0],
            [0.0 / 1.5, 0.0 / 1e-6, 4.0 / 3.0],
        ]
    )

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_mean_act_normalization_single_row() -> None:
    # Test with single row
    X_sae = torch.tensor([[1.0, 2.0, 0.0, -3.0]])

    result = mean_act_normalization(X_sae)

    # Each column has at most one non-zero value, so means are just the abs values
    expected = torch.tensor([[1.0 / 1.0, 2.0 / 2.0, 0.0 / 1e-6, -3.0 / 3.0]])

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_mean_act_normalization_single_column() -> None:
    # Test with single column
    X_sae = torch.tensor([[1.0], [0.0], [2.0], [-1.0]])

    result = mean_act_normalization(X_sae)

    # Column has values [1.0, 0.0, 2.0, -1.0], abs sum = 4.0, non-zero count = 3, mean = 4.0/3
    expected = torch.tensor(
        [[1.0 / (4.0 / 3)], [0.0 / (4.0 / 3)], [2.0 / (4.0 / 3)], [-1.0 / (4.0 / 3)]]
    )

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_get_sorted_indices_without_normalization() -> None:
    X_train_sae = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 0.0, 5.0],
            [0.0, 1.0, 4.0],
        ]
    )
    y_train = torch.tensor([0, 0, 1, 1])

    sorted_indices = get_sorted_indices(X_train_sae, y_train, "raw_mean_diff")

    class_0_mean = torch.tensor([1.5, 2.5, 0.5])
    class_1_mean = torch.tensor([0.0, 0.5, 4.5])
    diff = torch.abs(class_1_mean - class_0_mean)

    expected_order = torch.argsort(diff, descending=True)

    torch.testing.assert_close(sorted_indices, expected_order)


def test_get_sorted_indices_with_mean_normalization() -> None:
    X_train_sae = torch.tensor(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 1.0],
            [0.0, 0.0, 100.0],
            [0.0, 5.0, 200.0],
        ]
    )
    y_train = torch.tensor([0, 0, 1, 1])

    sorted_indices = get_sorted_indices(X_train_sae, y_train, "mean_diff")
    assert torch.equal(sorted_indices, get_sorted_indices(X_train_sae, y_train))

    X_normalized = mean_act_normalization(X_train_sae)
    class_0_mean = X_normalized[y_train == 0].mean(dim=0)
    class_1_mean = X_normalized[y_train == 1].mean(dim=0)
    diff = torch.abs(class_1_mean - class_0_mean)
    expected_order = torch.argsort(diff, descending=True)

    torch.testing.assert_close(sorted_indices, expected_order)


def test_get_sorted_indices_with_custom_normalization() -> None:
    X_train_sae = torch.tensor(
        [
            [1.0, 5.0, 100.0],
            [2.0, 6.0, 200.0],
            [10.0, 6.0, 105.0],
            [11.0, 7.0, 110.0],
        ]
    )
    y_train = torch.tensor([0, 0, 1, 1])

    def keep_first_dim_only(X: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(X)
        result[:, 0] = X[:, 0]
        return result

    sorted_indices = get_sorted_indices(
        X_train_sae, y_train, mean_diff_selection_fn(keep_first_dim_only)
    )

    expected_indices = torch.tensor([0, 1, 2])

    torch.testing.assert_close(sorted_indices, expected_indices)

    sorted_indices_without_norm = get_sorted_indices(
        X_train_sae, y_train, "raw_mean_diff"
    )
    expected_indices_without_norm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        sorted_indices_without_norm, expected_indices_without_norm
    )


def test_get_sorted_indices_different_normalizations_produce_different_results() -> (
    None
):
    X_train_sae = torch.tensor(
        [
            [1.0, 100.0, 0.0],
            [2.0, 200.0, 1.0],
            [0.0, 0.0, 50.0],
            [0.0, 10.0, 60.0],
        ]
    )
    y_train = torch.tensor([0, 0, 1, 1])

    indices_without_norm = get_sorted_indices(X_train_sae, y_train, "raw_mean_diff")
    indices_with_norm = get_sorted_indices(X_train_sae, y_train, "mean_diff")

    assert not torch.equal(indices_without_norm, indices_with_norm)


def test_get_sorted_indices_sorts_by_absolute_difference() -> None:
    X_train_sae = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -3.0, 2.0, 5.0],
            [1.0, -3.0, 2.0, 5.0],
        ]
    )
    y_train = torch.tensor([0, 0, 1, 1])

    sorted_indices = get_sorted_indices(X_train_sae, y_train, "raw_mean_diff")

    expected_indices = torch.tensor([3, 1, 2, 0])

    torch.testing.assert_close(sorted_indices, expected_indices)


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_eval_normal_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "normal"
    batch_size: int = 32

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    success: bool = run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        device="cpu",
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
        ks=[1, 2, 4, 8],
    )
    assert success

    expected_save_path: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path,
    )
    assert expected_save_path.exists(), f"Expected file not found: {expected_save_path}"

    with open(expected_save_path) as f:
        results: list[dict] = json.load(f)

    assert isinstance(results, list)
    assert len(results) == 4
    for item in results:
        assert "k" in item
        assert "dataset" in item
        assert item["dataset"] == TEST_DATASET_NAME
        assert "hook_name" in item
        assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
        assert "reg_type" in item
        assert item["reg_type"] == reg_type
        assert "binarize" in item
        assert not item["binarize"]
        assert "test_f1" in item
        assert item["test_f1"] >= 0.1
        assert "test_acc" in item
        assert item["test_acc"] >= 0.1
        assert "test_auc" in item
        assert item["test_auc"] >= 0.1
        assert "val_auc" in item
        assert item["val_auc"] >= 0.1


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_eval_scarcity_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "scarcity"
    batch_size: int = 32
    num_train: int = 50  # Small number for testing scarcity

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    success: bool = run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        device="cpu",
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
        ks=[1, 2],  # Smaller k list for faster test
        num_train=num_train,
    )
    assert success

    expected_save_path: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path,
        num_train=num_train,
    )
    assert expected_save_path.exists(), f"Expected file not found: {expected_save_path}"

    with open(expected_save_path) as f:
        results: list[dict] = json.load(f)

    assert isinstance(results, list)
    assert len(results) == 2  # Matches the number of ks
    for item in results:
        assert "k" in item
        assert "dataset" in item
        assert item["dataset"] == TEST_DATASET_NAME
        assert "hook_name" in item
        assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
        assert "reg_type" in item
        assert item["reg_type"] == reg_type
        assert "binarize" in item
        assert not item["binarize"]
        assert "test_f1" in item
        assert item["test_f1"] >= 0.0  # Looser bound for scarcity
        assert "test_acc" in item
        assert item["test_acc"] >= 0.0  # Looser bound for scarcity
        assert "test_auc" in item
        assert item["test_auc"] >= 0.0  # Looser bound for scarcity
        assert "val_auc" in item
        assert item["val_auc"] >= 0.0  # Looser bound for scarcity
        assert "num_train" in item
        assert item["num_train"] == num_train


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_eval_imbalance_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "imbalance"
    batch_size: int = 32
    frac: float = 0.1  # Small fraction for testing imbalance

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    success: bool = run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        device="cpu",
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
        ks=[1, 2],  # Smaller k list for faster test
        frac=frac,
    )
    assert success

    expected_save_path: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path,
        frac=frac,
    )
    assert expected_save_path.exists(), f"Expected file not found: {expected_save_path}"

    with open(expected_save_path) as f:
        results: list[dict] = json.load(f)

    assert isinstance(results, list)
    assert len(results) == 2  # Matches the number of ks
    for item in results:
        assert "k" in item
        assert "dataset" in item
        assert item["dataset"] == TEST_DATASET_NAME
        assert "hook_name" in item
        assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
        assert "reg_type" in item
        assert item["reg_type"] == reg_type
        assert "binarize" in item
        assert not item["binarize"]
        assert "test_f1" in item
        assert item["test_f1"] >= 0.0  # Looser bound for imbalance
        assert "test_acc" in item
        assert item["test_acc"] >= 0.0  # Looser bound for imbalance
        assert "test_auc" in item
        assert item["test_auc"] >= 0.0  # Looser bound for imbalance
        assert "val_auc" in item
        assert item["val_auc"] >= 0.0  # Looser bound for imbalance
        assert "frac" in item
        assert item["frac"] == frac


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_evals_normal_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "normal"

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    run_sae_evals(
        sae=gpt2_l4_sae,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        ks=[1, 2, 4, 8],
        datasets=[TEST_DATASET_NAME],
        device="cpu",
    )

    expected_save_path: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path,
    )
    assert expected_save_path.exists(), f"Expected file not found: {expected_save_path}"

    with open(expected_save_path) as f:
        results: list[dict] = json.load(f)

    assert isinstance(results, list)
    assert len(results) == 4
    for item in results:
        assert "k" in item
        assert "dataset" in item
        assert item["dataset"] == TEST_DATASET_NAME
        assert "hook_name" in item
        assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
        assert "reg_type" in item
        assert item["reg_type"] == reg_type
        assert "binarize" in item
        assert not item["binarize"]
        assert "test_f1" in item
        assert item["test_f1"] >= 0.1
        assert "test_acc" in item
        assert item["test_acc"] >= 0.1
        assert "test_auc" in item
        assert item["test_auc"] >= 0.1
        assert "val_auc" in item
        assert item["val_auc"] >= 0.1


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_evals_scarcity_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "scarcity"

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    run_sae_evals(
        sae=gpt2_l4_sae,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        ks=[1, 2],  # Smaller k list for faster test
        datasets=[TEST_DATASET_NAME],
        device="cpu",
    )

    # For scarcity setting, run_sae_evals creates multiple files for different num_train values
    # We'll check that at least one file exists and has the expected structure
    train_sizes = get_training_sizes()
    dataset_sizes = get_dataset_sizes()

    # Find valid training sizes for this dataset
    valid_train_sizes = [
        size for size in train_sizes if size <= dataset_sizes[TEST_DATASET_NAME] - 100
    ]
    assert len(valid_train_sizes) > 0, "No valid training sizes found"

    # Check that at least one result file was created
    found_valid_result = False
    for num_train in valid_train_sizes:
        expected_save_path: Path = get_save_metrics_path(
            dataset=TEST_DATASET_NAME,
            hook_name=f"blocks.{layer}.hook_resid_post",
            reg_type=reg_type,
            model_name=model_name,
            setting=setting,
            sae_results_path=sae_results_path,
            num_train=num_train,
        )
        if expected_save_path.exists():
            found_valid_result = True
            with open(expected_save_path) as f:
                results: list[dict] = json.load(f)

            assert isinstance(results, list)
            assert len(results) == 2  # Matches the number of ks
            for item in results:
                assert "k" in item
                assert "dataset" in item
                assert item["dataset"] == TEST_DATASET_NAME
                assert "hook_name" in item
                assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
                assert "reg_type" in item
                assert item["reg_type"] == reg_type
                assert "binarize" in item
                assert not item["binarize"]
                assert "test_f1" in item
                assert item["test_f1"] >= 0.0  # Looser bound for scarcity
                assert "test_acc" in item
                assert item["test_acc"] >= 0.0  # Looser bound for scarcity
                assert "test_auc" in item
                assert item["test_auc"] >= 0.0  # Looser bound for scarcity
                assert "val_auc" in item
                assert item["val_auc"] >= 0.0  # Looser bound for scarcity
                assert "num_train" in item
                assert item["num_train"] == num_train
            break

    assert found_valid_result, "No valid result files found for scarcity setting"


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_run_sae_evals_imbalance_setting(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer, reg_type: RegType
) -> None:
    sae_results_path = tmp_path / "sae_cache"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "imbalance"

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    run_sae_evals(
        sae=gpt2_l4_sae,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        results_path=sae_results_path,
        model_cache_path=model_cache_path,
        ks=[1, 2],  # Smaller k list for faster test
        datasets=[TEST_DATASET_NAME],
        device="cpu",
    )

    # For imbalance setting, run_sae_evals creates multiple files for different frac values
    # We'll check that at least one file exists and has the expected structure
    fracs = get_class_imbalance()

    # Check that at least one result file was created
    found_valid_result = False
    for frac in fracs:
        expected_save_path: Path = get_save_metrics_path(
            dataset=TEST_DATASET_NAME,
            hook_name=f"blocks.{layer}.hook_resid_post",
            reg_type=reg_type,
            model_name=model_name,
            setting=setting,
            sae_results_path=sae_results_path,
            frac=frac,
        )
        if expected_save_path.exists():
            found_valid_result = True
            with open(expected_save_path) as f:
                results: list[dict] = json.load(f)

            assert isinstance(results, list)
            assert len(results) == 2  # Matches the number of ks
            for item in results:
                assert "k" in item
                assert "dataset" in item
                assert item["dataset"] == TEST_DATASET_NAME
                assert "hook_name" in item
                assert item["hook_name"] == f"blocks.{layer}.hook_resid_post"
                assert "reg_type" in item
                assert item["reg_type"] == reg_type
                assert "binarize" in item
                assert not item["binarize"]
                assert "test_f1" in item
                assert item["test_f1"] >= 0.0  # Looser bound for imbalance
                assert "test_acc" in item
                assert item["test_acc"] >= 0.0  # Looser bound for imbalance
                assert "test_auc" in item
                assert item["test_auc"] >= 0.0  # Looser bound for imbalance
                assert "val_auc" in item
                assert item["val_auc"] >= 0.0  # Looser bound for imbalance
                assert "frac" in item
                assert item["frac"] == frac
            break

    assert found_valid_result, "No valid result files found for imbalance setting"


def test_run_sae_eval_different_selection_methods_produce_different_indices(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer
) -> None:
    sae_results_path_mean = tmp_path / "sae_cache_mean"
    sae_results_path_none = tmp_path / "sae_cache_none"
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    model_name: str = "gpt2"
    setting: Setting = "normal"
    batch_size: int = 32
    reg_type: RegType = "l1"

    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])

    success_mean: bool = run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        device="cpu",
        results_path=sae_results_path_mean,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
        ks=[16],
        selection_method="mean_diff",
    )
    assert success_mean

    success_none: bool = run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        setting=setting,
        model_name=model_name,
        device="cpu",
        results_path=sae_results_path_none,
        model_cache_path=model_cache_path,
        batch_size=batch_size,
        ks=[16],
        selection_method="raw_mean_diff",
    )
    assert success_none

    save_path_mean: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path_mean,
    )
    save_path_none: Path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type=reg_type,
        model_name=model_name,
        setting=setting,
        sae_results_path=sae_results_path_none,
        selection_method="raw_mean_diff",
    )

    with open(save_path_mean) as f:
        results_mean: list[dict] = json.load(f)
    with open(save_path_none) as f:
        results_none: list[dict] = json.load(f)

    indices_mean = results_mean[0]["indices"]
    indices_none = results_none[0]["indices"]

    assert indices_mean != indices_none


def test_run_sae_eval_binarize_thresholds_at_nonzero(
    gpt2_l4_sae: SAE,
    tmp_path: Path,
    gpt2_model: HookedTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_cache_path = tmp_path / "model_cache"
    layer: int = 4
    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])

    captured_activations = []
    captured_probe_inputs = []

    def capture_activations(**kwargs):
        activations = generate_sae_activations(**kwargs)
        captured_activations.append(activations)
        return activations

    def capture_probe_inputs(**kwargs):
        captured_probe_inputs.append(kwargs)
        return find_best_reg(**kwargs)

    monkeypatch.setattr(
        run_sae_evals_module, "generate_sae_activations", capture_activations
    )
    monkeypatch.setattr(run_sae_evals_module, "find_best_reg", capture_probe_inputs)

    run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=f"blocks.{layer}.hook_resid_post",
        reg_type="l2",
        setting="normal",
        model_name="gpt2",
        device="cpu",
        results_path=tmp_path / "sae_cache",
        model_cache_path=model_cache_path,
        batch_size=32,
        ks=[16],
        binarize=True,
    )

    activations = captured_activations[0]
    probe_inputs = captured_probe_inputs[0]
    # latents are ranked on the binarized activations the probe is trained on
    indices = get_sorted_indices(activations.X_train != 0, activations.y_train)[:16]
    indices_from_continuous = get_sorted_indices(
        activations.X_train, activations.y_train
    )[:16]
    assert indices.tolist() != indices_from_continuous.tolist()
    raw_train = activations.X_train[:, indices]
    raw_test = activations.X_test[:, indices]

    # make sure there are small nonzero activations that a higher threshold would drop
    assert ((raw_train > 0) & (raw_train <= 1)).any()
    torch.testing.assert_close(probe_inputs["X_train"], raw_train != 0)
    torch.testing.assert_close(probe_inputs["X_test"], raw_test != 0)


@pytest.mark.parametrize(
    "selection_method", ["correlation", "firing_rate", "auc", "accuracy"]
)
def test_run_sae_eval_selection_method_ranks_latents_on_probe_inputs(
    gpt2_l4_sae: SAE,
    tmp_path: Path,
    gpt2_model: HookedTransformer,
    monkeypatch: pytest.MonkeyPatch,
    selection_method: SelectionMethod,
) -> None:
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "sae_cache"
    layer: int = 4
    hook_name = f"blocks.{layer}.hook_resid_post"
    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])

    captured_activations = []
    captured_probe_inputs = []

    def capture_activations(**kwargs):
        activations = generate_sae_activations(**kwargs)
        captured_activations.append(activations)
        return activations

    def capture_probe_inputs(**kwargs):
        captured_probe_inputs.append(kwargs)
        return find_best_reg(**kwargs)

    monkeypatch.setattr(
        run_sae_evals_module, "generate_sae_activations", capture_activations
    )
    monkeypatch.setattr(run_sae_evals_module, "find_best_reg", capture_probe_inputs)

    for binarize in (False, True):
        run_sae_eval(
            sae=gpt2_l4_sae,
            dataset=TEST_DATASET_NAME,
            hook_name=hook_name,
            reg_type="l2",
            setting="normal",
            model_name="gpt2",
            device="cpu",
            results_path=results_path,
            model_cache_path=model_cache_path,
            batch_size=32,
            ks=[4],
            binarize=binarize,
            selection_method=selection_method,
        )

    activations = captured_activations[0]
    # continuous probes rank the continuous activations
    indices = get_sorted_indices(
        activations.X_train, activations.y_train, selection_method
    )[:4]
    torch.testing.assert_close(
        captured_probe_inputs[0]["X_train"], activations.X_train[:, indices]
    )
    # binarized probes rank the binarized activations the probe is trained on
    indices_binarized = get_sorted_indices(
        activations.X_train != 0, activations.y_train, selection_method
    )[:4]
    torch.testing.assert_close(
        captured_probe_inputs[1]["X_train"],
        activations.X_train[:, indices_binarized] != 0,
    )

    for binarize in (False, True):
        save_path = get_save_metrics_path(
            dataset=TEST_DATASET_NAME,
            hook_name=hook_name,
            reg_type="l2",
            model_name="gpt2",
            sae_results_path=results_path,
            binarize=binarize,
            selection_method=selection_method,
        )
        assert save_path.name.endswith(f"_sel-{selection_method}.json")
        with open(save_path) as f:
            results = json.load(f)
        assert [item["selection_method"] for item in results] == [selection_method]
        assert [item["binarize"] for item in results] == [binarize]
    # the default method's result file is not written
    assert not get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=hook_name,
        reg_type="l2",
        model_name="gpt2",
        sae_results_path=results_path,
    ).exists()


def test_run_sae_eval_default_selection_is_recorded_as_mean_diff(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer
) -> None:
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "sae_cache"
    layer: int = 4
    hook_name = f"blocks.{layer}.hook_resid_post"
    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    run_sae_eval(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=hook_name,
        reg_type="l2",
        setting="normal",
        model_name="gpt2",
        device="cpu",
        results_path=results_path,
        model_cache_path=model_cache_path,
        batch_size=32,
        ks=[2],
    )
    save_path = get_save_metrics_path(
        dataset=TEST_DATASET_NAME,
        hook_name=hook_name,
        reg_type="l2",
        model_name="gpt2",
        sae_results_path=results_path,
    )
    with open(save_path) as f:
        results = json.load(f)
    assert results[0]["selection_method"] == "mean_diff"


def test_run_sae_eval_deprecated_mean_diff_normalization_maps_to_selection_method(
    gpt2_l4_sae: SAE, tmp_path: Path, gpt2_model: HookedTransformer
) -> None:
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "sae_cache"
    layer: int = 4
    hook_name = f"blocks.{layer}.hook_resid_post"
    generate_model_activations(gpt2_model, model_cache_path, layers=[layer])
    common = dict(
        sae=gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name=hook_name,
        reg_type="l2",
        setting="normal",
        model_name="gpt2",
        device="cpu",
        model_cache_path=model_cache_path,
        batch_size=32,
        ks=[16],
    )
    with pytest.warns(DeprecationWarning, match="mean_diff_normalization"):
        run_sae_eval(
            **common,  # type: ignore[arg-type]
            results_path=results_path / "deprecated",
            mean_diff_normalization="none",
        )
    run_sae_eval(
        **common,  # type: ignore[arg-type]
        results_path=results_path / "new",
        selection_method="raw_mean_diff",
    )
    results = []
    for folder in ("deprecated", "new"):
        save_path = get_save_metrics_path(
            dataset=TEST_DATASET_NAME,
            hook_name=hook_name,
            reg_type="l2",
            model_name="gpt2",
            sae_results_path=results_path / folder,
            selection_method="raw_mean_diff",
        )
        with open(save_path) as f:
            results.append(json.load(f))
    assert results[0][0]["indices"] == results[1][0]["indices"]
    assert results[0][0]["selection_method"] == "raw_mean_diff"


def test_resolve_selection_method() -> None:
    assert resolve_selection_method() == "mean_diff"
    assert resolve_selection_method("auc") == "auc"
    with pytest.warns(DeprecationWarning):
        assert resolve_selection_method("mean_diff", "mean") == "mean_diff"
    with pytest.warns(DeprecationWarning):
        assert resolve_selection_method("mean_diff", "none") == "raw_mean_diff"

    def double(X: torch.Tensor) -> torch.Tensor:
        return 2 * X

    with pytest.warns(DeprecationWarning):
        selection_fn = resolve_selection_method("mean_diff", double)
    assert callable(selection_fn)
    X = torch.tensor([[0.0, 1.0], [0.0, 3.0], [4.0, 1.0], [2.0, 1.0]])
    y = torch.tensor([0, 0, 1, 1])
    torch.testing.assert_close(selection_fn(X, y), torch.tensor([6.0, 2.0]))

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        resolve_selection_method("auc", "mean")
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="Invalid mean_diff_normalization"),
    ):
        resolve_selection_method("mean_diff", "bad")  # type: ignore[arg-type]
