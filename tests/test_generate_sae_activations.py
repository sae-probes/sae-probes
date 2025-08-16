from pathlib import Path

from sae_lens import SAE, HookedSAETransformer

from sae_probes.generate_sae_activations import (
    generate_sae_activations_imbalance,
    generate_sae_activations_normal,
    generate_sae_activations_scarcity,
)
from tests.helpers import TEST_DATASET_NAME, generate_model_activations


def test_generate_sae_activations_normal(
    gpt2_model: HookedSAETransformer, tmp_path: Path, gpt2_l4_sae: SAE
):
    model_cache_path = tmp_path / "model_cache"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    sae_acts = generate_sae_activations_normal(
        gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name="blocks.4.hook_resid_post",
        model_name="gpt2",
        device="cpu",
        model_cache_path=model_cache_path,
    )
    assert sae_acts.X_train.shape == (900, 24576)
    assert sae_acts.y_train.shape == (900,)
    # this seems odd, why 99 instead of 100??
    assert sae_acts.X_test.shape == (99, 24576)
    assert sae_acts.y_test.shape == (99,)


def test_generate_sae_activations_imbalance(
    gpt2_model: HookedSAETransformer, tmp_path: Path, gpt2_l4_sae: SAE
):
    model_cache_path = tmp_path / "model_cache"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    sae_acts = generate_sae_activations_imbalance(
        gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name="blocks.4.hook_resid_post",
        frac=0.5,
        model_name="gpt2",
        device="cpu",
        model_cache_path=model_cache_path,
    )
    assert sae_acts.X_train.shape == (426, 24576)
    assert sae_acts.y_train.shape == (426,)
    assert sae_acts.X_test.shape == (100, 24576)
    assert sae_acts.y_test.shape == (100,)


def test_generate_sae_activations_scarsity(
    gpt2_model: HookedSAETransformer, tmp_path: Path, gpt2_l4_sae: SAE
):
    model_cache_path = tmp_path / "model_cache"
    generate_model_activations(gpt2_model, model_cache_path, layers=[4])
    sae_acts = generate_sae_activations_scarcity(
        gpt2_l4_sae,
        dataset=TEST_DATASET_NAME,
        hook_name="blocks.4.hook_resid_post",
        num_train=123,
        model_name="gpt2",
        device="cpu",
        model_cache_path=model_cache_path,
    )
    assert sae_acts.X_train.shape == (123, 24576)
    assert sae_acts.y_train.shape == (123,)
    assert sae_acts.X_test.shape == (876, 24576)
    assert sae_acts.y_test.shape == (876,)
