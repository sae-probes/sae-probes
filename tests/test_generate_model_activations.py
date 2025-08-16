from pathlib import Path

import torch
from sae_lens import HookedSAETransformer

from sae_probes.constants import DATA_PATH
from sae_probes.generate_model_activations import (
    _get_text_lengths,
    _process_activations,
    generate_single_dataset_activations,
)
from sae_probes.utils_hooks import get_layer_from_hook_name


def test_process_activations_gives_same_results_regardless_of_batch_size(
    gpt2_model: HookedSAETransformer,
):
    texts = [
        "Hello, world!",
        "this is some longer text",
        "short",
        "he he he 1 2 3 4 5 6 7 8 9 10",
    ]
    max_seq_len = 1024
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in [2, 3, 4]]

    activations_batched = _process_activations(
        model=gpt2_model,
        texts=texts,
        batch_size=3,
        max_seq_len=max_seq_len,
        hook_names=hook_names,
        max_layer=4,
        device="cpu",
    )
    activations_unbatched = _process_activations(
        model=gpt2_model,
        texts=texts,
        batch_size=1,
        max_seq_len=max_seq_len,
        hook_names=hook_names,
        max_layer=4,
        device="cpu",
    )

    _, test_act = gpt2_model.run_with_cache(
        "Hello, world!",
        names_filter=hook_names,
        stop_at_layer=5,
        prepend_bos=False,
    )
    assert torch.allclose(
        activations_batched["blocks.2.hook_resid_post"][0],
        test_act["blocks.2.hook_resid_post"][0, -1, :],
        atol=1e-5,
    )

    for batched_acts, unbatched_acts in zip(
        activations_batched.values(), activations_unbatched.values()
    ):
        assert torch.allclose(batched_acts, unbatched_acts, atol=1e-5)


def test_get_layer_from_hook_name_parsing():
    assert get_layer_from_hook_name("blocks.0.hook_resid_post") == 0
    assert get_layer_from_hook_name("blocks.12.hook_mlp_out") == 12
    assert get_layer_from_hook_name("hook_embed") is None
    assert get_layer_from_hook_name("final_ln.hook_normalized") is None
    assert get_layer_from_hook_name("") is None
    assert get_layer_from_hook_name("blocks.notint.hook_resid_post") is None


def test_get_text_lengths(gpt2_model: HookedSAETransformer):
    texts = [
        "Hello, world!",
        "this is some longer text",
        "short",
        "he he he 1 2 3 4 5 6 7 8 9 10",
    ]
    text_lengths = _get_text_lengths(gpt2_model, texts)
    assert text_lengths == [4, 5, 1, 13]


def test_generate_single_dataset_activations(
    gpt2_model: HookedSAETransformer, tmp_path: Path
):
    # this is relatively small dataset, so it's fine to run this fully on CPU in CI
    generate_single_dataset_activations(
        model=gpt2_model,
        model_name="gpt2",
        dataset_path=str(DATA_PATH / "cleaned_data" / "119_us_state_TX.csv.zst"),
        hook_names=["blocks.1.hook_resid_post", "blocks.2.hook_resid_post"],
        model_cache_path=tmp_path,
        device="cpu",
    )
    for layer in [1, 2]:
        assert (
            tmp_path
            / "model_activations_gpt2"
            / f"119_us_state_TX_blocks.{layer}.hook_resid_post.pt"
        ).exists()
        acts = torch.load(
            tmp_path
            / "model_activations_gpt2"
            / f"119_us_state_TX_blocks.{layer}.hook_resid_post.pt"
        )
        assert acts.shape == (1000, 768)
