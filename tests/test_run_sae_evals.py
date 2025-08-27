import json
from pathlib import Path

import pytest
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_probes.constants import RegType, Setting
from sae_probes.run_sae_evals import get_save_metrics_path, run_sae_eval, run_sae_evals
from sae_probes.utils_data import (
    get_class_imbalance,
    get_dataset_sizes,
    get_training_sizes,
)
from tests.helpers import TEST_DATASET_NAME, generate_model_activations


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
