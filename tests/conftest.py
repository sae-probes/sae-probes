import pytest
from sae_lens import SAE
from transformer_lens import HookedTransformer


@pytest.fixture(scope="session")
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_tokenizer(gpt2_model: HookedTransformer):
    return gpt2_model.tokenizer


# function-scoped: some tests change the SAE in place (e.g. its dtype)
@pytest.fixture
def gpt2_l4_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.4.hook_resid_pre", device="cpu"
    )[0]
