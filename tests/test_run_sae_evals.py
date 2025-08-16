import json
from pathlib import Path

import pytest
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_probes.constants import RegType, Setting
from sae_probes.run_sae_evals import get_save_metrics_path, run_sae_eval
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
