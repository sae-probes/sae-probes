import shutil
import tempfile
from pathlib import Path

from transformer_lens import HookedTransformer

from sae_probes.constants import DATA_PATH
from sae_probes.generate_model_activations import generate_single_dataset_activations

TEST_DATASET_NAME = "119_us_state_TX"
TEST_DATASET_PATH = DATA_PATH / "cleaned_data" / f"{TEST_DATASET_NAME}.csv.zst"


# generating GPT-2 activations is the slowest part of most tests, so each distinct request
# is computed once per test session and copied into the directory the test asked for
_ACTIVATIONS_CACHE: dict[tuple[tuple[int, ...], str], Path] = {}


def generate_model_activations(
    model: HookedTransformer,
    model_cache_path: Path,
    layers: list[int],
    dataset_path: Path = TEST_DATASET_PATH,
) -> dict[int, Path]:
    key = (tuple(layers), str(dataset_path))
    if key not in _ACTIVATIONS_CACHE:
        cache_dir = Path(tempfile.mkdtemp(prefix="sae_probes_test_activations_"))
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        generate_single_dataset_activations(
            model=model,
            model_name="gpt2",
            dataset_path=dataset_path,
            hook_names=hook_names,
            model_cache_path=cache_dir,
            device="cpu",
        )
        _ACTIVATIONS_CACHE[key] = cache_dir
    shutil.copytree(_ACTIVATIONS_CACHE[key], model_cache_path, dirs_exist_ok=True)
    return {
        layer: model_cache_path
        / "model_activations_gpt2"
        / f"119_us_state_TX_blocks.{layer}.hook_resid_post.pt"
        for layer in layers
    }
