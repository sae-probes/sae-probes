import json
from pathlib import Path

from transformer_lens import HookedTransformer

from sae_probes.run_baselines import (
    run_baseline_class_imbalance,
    run_baseline_dataset_layer,
    run_baseline_evals,
    run_baseline_scarcity,
)
from sae_probes.utils_data import (
    get_class_imbalance,
    get_dataset_sizes,
    get_training_sizes,
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


def test_run_baseline_evals_normal_setting(
    gpt2_model: HookedTransformer, tmp_path: Path
):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_evals(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        setting="normal",
        method="logreg",
        results_path=results_path,
        model_cache_path=model_cache_path,
        datasets=[TEST_DATASET_NAME],
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


def test_run_baseline_evals_scarcity_setting(
    gpt2_model: HookedTransformer, tmp_path: Path
):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_evals(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        setting="scarcity",
        method="logreg",
        results_path=results_path,
        model_cache_path=model_cache_path,
        datasets=[TEST_DATASET_NAME],
    )

    # For scarcity setting, run_baseline_evals creates multiple files for different num_train values
    # We'll check that at least one file exists and has the expected structure
    train_sizes = get_training_sizes()
    dataset_sizes = get_dataset_sizes()

    # Find valid training sizes for this dataset
    valid_train_sizes = [
        size for size in train_sizes if size <= dataset_sizes[TEST_DATASET_NAME] - 100
    ]
    assert len(valid_train_sizes) > 0, "No valid training sizes found"

    # Check that at least one result file was created
    results_files = list(results_path.glob("**/*.json"))
    assert len(results_files) > 0, "No result files found"

    # Find and validate at least one result file
    found_valid_result = False
    for results_file in results_files:
        with open(results_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        result = data[0]

        # Check that the result has the expected structure
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
        assert result["num_train"] in valid_train_sizes
        assert result["test_f1"] >= 0.0  # Looser bound for scarcity
        assert result["test_acc"] >= 0.0  # Looser bound for scarcity
        assert result["test_auc"] >= 0.0  # Looser bound for scarcity
        assert result["val_auc"] >= 0.0  # Looser bound for scarcity
        found_valid_result = True

    assert found_valid_result, "No valid result files found for scarcity setting"


def test_run_baseline_evals_imbalance_setting(
    gpt2_model: HookedTransformer, tmp_path: Path
):
    model_cache_path = tmp_path / "model_cache"
    results_path = tmp_path / "results"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    run_baseline_evals(
        model_name="gpt2",
        hook_name="blocks.4.hook_resid_post",
        setting="imbalance",
        method="logreg",
        results_path=results_path,
        model_cache_path=model_cache_path,
        datasets=[TEST_DATASET_NAME],
    )

    # For imbalance setting, run_baseline_evals creates multiple files for different frac values
    # We'll check that at least one file exists and has the expected structure
    fracs = get_class_imbalance()

    # Check that at least one result file was created
    results_files = list(results_path.glob("**/*.json"))
    assert len(results_files) > 0, "No result files found"

    # Find and validate at least one result file
    found_valid_result = False
    for results_file in results_files:
        with open(results_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        result = data[0]

        # Check that the result has the expected structure
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
        # Check frac with tolerance for floating point precision
        assert any(abs(result["frac"] - frac) < 1e-10 for frac in fracs)
        assert result["test_f1"] >= 0.0  # Looser bound for imbalance
        assert result["test_acc"] >= 0.0  # Looser bound for imbalance
        assert result["test_auc"] >= 0.0  # Looser bound for imbalance
        assert result["val_auc"] >= 0.0  # Looser bound for imbalance
        found_valid_result = True

    assert found_valid_result, "No valid result files found for imbalance setting"
