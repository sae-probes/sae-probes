from pathlib import Path

from transformer_lens import HookedTransformer

from sae_probes.constants import DATA_PATH
from sae_probes.generate_model_activations import generate_single_dataset_activations

TEST_DATASET_NAME = "119_us_state_TX"
TEST_DATASET_PATH = DATA_PATH / "cleaned_data" / f"{TEST_DATASET_NAME}.csv.gz"


def generate_model_activations(
    model: HookedTransformer,
    model_cache_path: Path,
    layers: list[int],
    dataset_path: Path = TEST_DATASET_PATH,
) -> dict[int, Path]:
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    generate_single_dataset_activations(
        model=model,
        model_name="gpt2",
        dataset_path=dataset_path,
        hook_names=hook_names,
        model_cache_path=model_cache_path,
        device="cpu",
    )
    return {
        layer: model_cache_path
        / "model_activations_gpt2"
        / f"119_us_state_TX_blocks.{layer}.hook_resid_post.pt"
        for layer in layers
    }
