import json
from pathlib import Path

from transformer_lens import HookedTransformer

from sae_probes.run_baselines import (
    run_baseline_class_imbalance,
    run_baseline_dataset_layer,
    run_baseline_scarcity,
)
from tests.helpers import TEST_DATASET_NAME, generate_model_activations


def test_run_baseline_dataset_layer(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_dataset_layer(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.json"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/normal_setting/119_us_state_TX_blocks.4.hook_resid_post_logreg.json"
    )
    with open(results_files[0]) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    result = data[0]
    assert set(result.keys()) >= {
        "dataset",
        "hook_name",
        "method",
        "num_train",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }
    assert result["dataset"] == "119_us_state_TX"
    assert result["hook_name"] == "blocks.4.hook_resid_post"
    assert result["method"] == "logreg"
    assert result["test_f1"] > 0.6
    assert result["test_acc"] > 0.6
    assert result["test_auc"] > 0.6
    assert result["val_auc"] > 0.6


def test_run_baseline_scarcity(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_scarcity(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        num_train=25,
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.json"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/scarcity_setting/119_us_state_TX_blocks.4.hook_resid_post_25_logreg.json"
    )
    with open(results_files[0]) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    result = data[0]
    assert set(result.keys()) >= {
        "dataset",
        "hook_name",
        "method",
        "num_train",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }
    assert result["dataset"] == "119_us_state_TX"
    assert result["hook_name"] == "blocks.4.hook_resid_post"
    assert result["method"] == "logreg"
    assert result["num_train"] == 25
    assert result["test_f1"] > 0.5
    assert result["test_acc"] > 0.5
    assert result["test_auc"] > 0.5
    assert result["val_auc"] > 0.5


def test_run_baseline_class_imbalance(gpt2_model: HookedTransformer, tmp_path: Path):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_class_imbalance(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        numbered_dataset=TEST_DATASET_NAME,
        method_name="logreg",
        dataset_frac=0.1,
        results_path=results_path,
        model_cache_path=model_cache_path,
    )
    results_files = list(results_path.glob("**/*.json"))
    assert len(results_files) == 1
    relative_path = results_files[0].relative_to(results_path)
    assert (
        str(relative_path)
        == "baseline_results_gpt2/imbalance_setting/119_us_state_TX_blocks.4.hook_resid_post_frac0.1_logreg.json"
    )
    with open(results_files[0]) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    result = data[0]
    assert set(result.keys()) >= {
        "dataset",
        "hook_name",
        "method",
        "num_train",
        "frac",
        "test_f1",
        "test_acc",
        "test_auc",
        "val_auc",
    }

    assert result["dataset"] == "119_us_state_TX"
    assert result["hook_name"] == "blocks.4.hook_resid_post"
    assert result["method"] == "logreg"
    assert result["num_train"] == 426
    assert result["frac"] == 0.1
    assert result["test_f1"] > 0.5
    assert result["test_acc"] > 0.5
    assert result["test_auc"] > 0.5
    assert result["val_auc"] > 0.5
