import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from sae_probes.latent_selection import (
    SELECTION_METHODS,
    accuracy_scores,
    auc_scores,
    correlation_scores,
    firing_rate_scores,
    forward_selection_order,
    get_selection_scores,
    get_sorted_indices,
    mean_act_normalization,
    mean_diff_scores,
    mean_diff_selection_fn,
    selection_method_name,
)
from sae_probes.run_sae_evals import get_save_metrics_path


def _sparse_activations(
    n_samples: int = 200, n_latents: int = 30, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-negative activations where most entries are exactly zero, like SAE codes."""
    generator = torch.Generator().manual_seed(seed)
    y = (torch.rand(n_samples, generator=generator) < 0.4).long()
    X = torch.rand(n_samples, n_latents, generator=generator)
    X = X * (torch.rand(n_samples, n_latents, generator=generator) < 0.3)
    # make some latents informative about the label
    X[:, 0] = X[:, 0] + 2.0 * y
    X[:, 1] = (0.5 + torch.rand(n_samples, generator=generator)) * (y == 0)
    return X, y


def _brute_force_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    best = max(float(y.mean()), 1 - float(y.mean()))
    for threshold in np.unique(x):
        pred = (x > threshold).astype(int)
        acc = float((pred == y).mean())
        best = max(best, acc, 1 - acc)
    return float(best)


def test_auc_scores_match_sklearn_on_sparse_activations() -> None:
    X, y = _sparse_activations()
    scores = auc_scores(X, y)
    for i in range(X.shape[1]):
        auc = roc_auc_score(y.numpy(), X[:, i].numpy())
        assert scores[i].item() == pytest.approx(max(auc, 1 - auc), abs=1e-6)


def test_auc_scores_match_sklearn_on_binarized_activations() -> None:
    X, y = _sparse_activations()
    X_binary = X != 0
    scores = auc_scores(X_binary, y)
    for i in range(X.shape[1]):
        auc = roc_auc_score(y.numpy(), X_binary[:, i].numpy().astype(float))
        assert scores[i].item() == pytest.approx(max(auc, 1 - auc), abs=1e-6)


def test_auc_scores_are_the_same_across_chunk_sizes() -> None:
    X, y = _sparse_activations()
    torch.testing.assert_close(auc_scores(X, y, chunk_size=7), auc_scores(X, y))
    torch.testing.assert_close(
        accuracy_scores(X, y, chunk_size=7), accuracy_scores(X, y)
    )


def test_auc_scores_ignore_monotonic_rescaling_of_a_latent() -> None:
    X, y = _sparse_activations()
    rescaled = X.clone()
    rescaled[:, 0] = rescaled[:, 0] ** 3 * 100
    rescaled[:, 2] = torch.log1p(rescaled[:, 2])
    torch.testing.assert_close(auc_scores(rescaled, y), auc_scores(X, y))
    torch.testing.assert_close(accuracy_scores(rescaled, y), accuracy_scores(X, y))


def test_auc_scores_constant_and_perfect_latents() -> None:
    y = torch.tensor([0, 0, 1, 1, 1])
    X = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.5, 2.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 9.0, 0.0],
        ]
    )
    scores = auc_scores(X, y)
    assert scores.tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_auc_scores_with_a_single_class_are_uninformative() -> None:
    X, _ = _sparse_activations()
    y = torch.ones(X.shape[0], dtype=torch.long)
    assert auc_scores(X, y).tolist() == pytest.approx([0.5] * X.shape[1])


def test_accuracy_scores_match_brute_force() -> None:
    X, y = _sparse_activations()
    scores = accuracy_scores(X, y)
    for i in range(X.shape[1]):
        expected = _brute_force_accuracy(X[:, i].numpy(), y.numpy())
        assert scores[i].item() == pytest.approx(expected, abs=1e-6)


def test_accuracy_scores_match_brute_force_on_binarized_activations() -> None:
    X, y = _sparse_activations()
    X_binary = X != 0
    scores = accuracy_scores(X_binary, y)
    for i in range(X.shape[1]):
        expected = _brute_force_accuracy(
            X_binary[:, i].numpy().astype(float), y.numpy()
        )
        assert scores[i].item() == pytest.approx(expected, abs=1e-6)


def test_accuracy_scores_constant_latent_gets_majority_class_accuracy() -> None:
    y = torch.tensor([0, 0, 0, 1])
    X = torch.zeros(4, 2)
    assert accuracy_scores(X, y).tolist() == pytest.approx([0.75, 0.75])


def test_mean_diff_scores_match_the_original_rule() -> None:
    X, y = _sparse_activations()
    X_normalized = mean_act_normalization(X)
    expected = (
        X_normalized[y == 1].mean(dim=0) - X_normalized[y == 0].mean(dim=0)
    ).abs()
    torch.testing.assert_close(mean_diff_scores(X, y, mean_act_normalization), expected)
    torch.testing.assert_close(get_selection_scores(X, y, "mean_diff"), expected)
    torch.testing.assert_close(get_selection_scores(X, y), expected)


def test_raw_mean_diff_scores_use_unnormalized_activations() -> None:
    X, y = _sparse_activations()
    expected = (X[y == 1].mean(dim=0) - X[y == 0].mean(dim=0)).abs()
    torch.testing.assert_close(get_selection_scores(X, y, "raw_mean_diff"), expected)


def test_firing_rate_scores_are_the_difference_in_firing_rates() -> None:
    X, y = _sparse_activations()
    active = (X != 0).float()
    expected = (active[y == 1].mean(dim=0) - active[y == 0].mean(dim=0)).abs()
    torch.testing.assert_close(firing_rate_scores(X, y), expected)
    torch.testing.assert_close(get_selection_scores(X, y, "firing_rate"), expected)
    # activation strength is ignored
    torch.testing.assert_close(firing_rate_scores(X * 100, y), expected)


def test_correlation_scores_match_numpy() -> None:
    X, y = _sparse_activations()
    X_np, y_np = X.numpy().astype(np.float64), y.numpy().astype(np.float64)
    expected = np.array(
        [abs(np.corrcoef(X_np[:, i], y_np)[0, 1]) for i in range(X_np.shape[1])]
    )
    np.testing.assert_allclose(correlation_scores(X, y).numpy(), expected, atol=1e-5)


def test_correlation_scores_rank_like_cohens_d() -> None:
    X, y = _sparse_activations()
    X_np, y_np = X.numpy().astype(np.float64), y.numpy()
    pos, neg = X_np[y_np == 1], X_np[y_np == 0]
    # pooled within-class variance, weighted by class size
    pooled_var = (len(pos) * pos.var(axis=0) + len(neg) * neg.var(axis=0)) / len(X_np)
    cohens_d = np.abs(pos.mean(axis=0) - neg.mean(axis=0)) / np.sqrt(pooled_var)
    assert (
        get_sorted_indices(X, y, "correlation").tolist()
        == np.argsort(-cohens_d, kind="stable").tolist()
    )


def test_correlation_scores_are_zero_for_constant_latents() -> None:
    y = torch.tensor([0, 0, 1, 1])
    X = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    assert correlation_scores(X, y).tolist() == [0.0, 0.0]


def test_magnitude_outliers_mislead_mean_diff_but_not_auc_or_accuracy() -> None:
    n = 100
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    # latent 0 fires on every example of class 1 and never on class 0
    clean = y.float() * 1.0
    # latent 1 fires weakly on everything, and very strongly on a few class 1 examples
    spiky = torch.full((2 * n,), 0.1)
    spiky[n : n + 5] = 500.0
    X = torch.stack([clean, spiky], dim=1)

    assert get_sorted_indices(X, y, "mean_diff")[0].item() == 1
    assert get_sorted_indices(X, y, "raw_mean_diff")[0].item() == 1
    for selection_method in (
        "correlation",
        "firing_rate",
        "auc",
        "accuracy",
        "forward",
    ):
        assert get_sorted_indices(X, y, selection_method)[0].item() == 0


@pytest.mark.parametrize("selection_method", SELECTION_METHODS)
def test_selection_scores_treat_both_directions_alike(selection_method) -> None:
    X, y = _sparse_activations()
    scores = get_selection_scores(X, y, selection_method)
    flipped = get_selection_scores(X, 1 - y, selection_method)
    torch.testing.assert_close(scores, flipped)
    assert scores.shape == (X.shape[1],)
    assert (scores >= 0).all()


def test_get_selection_scores_accepts_a_callable() -> None:
    X, y = _sparse_activations()

    def prefer_last_latents(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        return torch.arange(X.shape[1], dtype=torch.float32)

    scores = get_selection_scores(X, y, prefer_last_latents)
    assert scores.tolist() == list(range(X.shape[1]))
    indices = get_sorted_indices(X, y, prefer_last_latents)
    assert indices[:3].tolist() == [X.shape[1] - 1, X.shape[1] - 2, X.shape[1] - 3]
    assert selection_method_name(prefer_last_latents) == "prefer_last_latents"


def test_get_selection_scores_rejects_unknown_methods() -> None:
    X, y = _sparse_activations()
    with pytest.raises(ValueError, match="Invalid selection_method"):
        get_selection_scores(X, y, "not-a-method")  # type: ignore[arg-type]


def test_mean_diff_selection_fn_applies_a_custom_normalization() -> None:
    X, y = _sparse_activations()

    def double(X: torch.Tensor) -> torch.Tensor:
        return 2 * X

    selection_fn = mean_diff_selection_fn(double)
    torch.testing.assert_close(selection_fn(X, y), 2 * mean_diff_scores(X, y))
    torch.testing.assert_close(
        mean_diff_selection_fn(mean_act_normalization)(X, y),
        get_selection_scores(X, y, "mean_diff"),
    )


def test_get_sorted_indices_defaults_to_mean_diff() -> None:
    X, y = _sparse_activations()
    assert (
        get_sorted_indices(X, y).tolist()
        == get_sorted_indices(X, y, "mean_diff").tolist()
    )


@pytest.mark.parametrize(
    "selection_method", ["correlation", "firing_rate", "auc", "accuracy", "forward"]
)
def test_get_sorted_indices_puts_the_informative_latents_first(
    selection_method,
) -> None:
    X, y = _sparse_activations(n_samples=500)
    indices = get_sorted_indices(X, y, selection_method)
    assert set(indices[:2].tolist()) == {0, 1}


def _reference_forward_selection(
    X: np.ndarray, y: np.ndarray, n_select: int
) -> list[int]:
    """Orthogonal matching pursuit on the centered labels with a full refit each step."""
    Xs = X - X.mean(axis=0)
    Xs = Xs / np.linalg.norm(Xs, axis=0)
    target = y - y.mean()
    residual = target.copy()
    chosen: list[int] = []
    for _ in range(n_select):
        scores = np.abs(Xs.T @ residual)
        scores[chosen] = -1
        chosen.append(int(scores.argmax()))
        coef, *_ = np.linalg.lstsq(Xs[:, chosen], target, rcond=None)
        residual = target - Xs[:, chosen] @ coef
    return chosen


def test_forward_selection_matches_a_reference_implementation() -> None:
    X, y = _sparse_activations(n_samples=300, n_latents=40)
    expected = _reference_forward_selection(
        X.numpy().astype(np.float64), y.numpy().astype(np.float64), n_select=10
    )
    order = forward_selection_order(X, y, n_select=10)
    assert order[:10].tolist() == expected
    assert get_sorted_indices(X, y, "forward", k_max=10)[:10].tolist() == expected


def test_forward_selection_returns_every_latent_exactly_once() -> None:
    X, y = _sparse_activations()
    for n_select in (0, 1, 5, 1000):
        order = forward_selection_order(X, y, n_select=n_select)
        assert sorted(order.tolist()) == list(range(X.shape[1]))
    order = forward_selection_order(X, y, n_select=5, pool_size=8)
    assert sorted(order.tolist()) == list(range(X.shape[1]))


def test_forward_selection_first_latent_is_the_most_correlated() -> None:
    X, y = _sparse_activations()
    assert (
        get_sorted_indices(X, y, "forward")[0].item()
        == get_sorted_indices(X, y, "correlation")[0].item()
    )


def test_forward_selection_order_is_nested_in_the_number_of_steps() -> None:
    X, y = _sparse_activations(n_samples=300, n_latents=40)
    long_order = forward_selection_order(X, y, n_select=12)
    for n_select in (1, 3, 7):
        short_order = forward_selection_order(X, y, n_select=n_select)
        assert short_order[:n_select].tolist() == long_order[:n_select].tolist()
    assert (
        get_sorted_indices(X, y, "forward", k_max=3)[:3].tolist()
        == get_sorted_indices(X, y, "forward")[:3].tolist()
    )


def test_forward_selection_skips_redundant_latents() -> None:
    generator = torch.Generator().manual_seed(0)
    n = 400
    part_a = torch.rand(n, generator=generator) < 0.5
    part_b = torch.rand(n, generator=generator) < 0.3
    y = (part_a | part_b).long()
    noise = lambda: 0.01 * torch.rand(n, generator=generator)  # noqa: E731
    # latents 0-2 are near-copies of the same detector, latent 3 covers the rest
    X = torch.stack(
        [
            part_a.float() + noise(),
            part_a.float() + noise(),
            part_a.float() + noise(),
            (part_b & ~part_a).float() + noise(),
            torch.rand(n, generator=generator),
        ],
        dim=1,
    )
    # scoring latents one at a time picks two copies of the same detector
    for selection_method in ("correlation", "auc", "accuracy"):
        top2 = get_sorted_indices(X, y, selection_method)[:2].tolist()
        assert set(top2) <= {0, 1, 2}
    # forward selection picks one copy, then the complementary latent
    top2 = get_sorted_indices(X, y, "forward", k_max=2)[:2].tolist()
    assert top2[0] in {0, 1, 2}
    assert top2[1] == 3


def test_forward_selection_handles_duplicate_and_constant_latents() -> None:
    X, y = _sparse_activations(n_samples=120, n_latents=6)
    X = torch.cat([X, X[:, :1], torch.zeros(120, 2), torch.ones(120, 1)], dim=1)
    order = forward_selection_order(X, y, n_select=10)
    assert sorted(order.tolist()) == list(range(X.shape[1]))
    # the exact duplicate of latent 0 (index 6) is never selected before the others run out
    first_six = order[:6].tolist()
    assert not ({0, 6} <= set(first_six))


def test_forward_selection_stops_when_the_labels_are_fit_exactly() -> None:
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    X = torch.zeros(6, 5)
    X[:, 0] = y.float()  # perfectly predictive
    X[:, 1] = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    X[:, 2] = torch.tensor([0.3, 0.1, 0.9, 0.2, 0.4, 0.8])
    order = forward_selection_order(X, y, n_select=5)
    assert order[0].item() == 0
    # after the exact fit the rest follow in order of correlation with the label
    by_correlation = get_sorted_indices(X, y, "correlation").tolist()
    assert order[1:].tolist() == [i for i in by_correlation if i != 0]


def test_forward_selection_with_a_single_class_falls_back_to_correlation_order() -> (
    None
):
    X, _ = _sparse_activations()
    y = torch.ones(X.shape[0], dtype=torch.long)
    order = forward_selection_order(X, y)
    assert order.tolist() == get_sorted_indices(X, y, "correlation").tolist()


def test_forward_selection_works_on_binarized_activations() -> None:
    X, y = _sparse_activations(n_samples=300, n_latents=40)
    X_binary = X != 0
    expected = _reference_forward_selection(
        X_binary.numpy().astype(np.float64), y.numpy().astype(np.float64), n_select=6
    )
    assert forward_selection_order(X_binary, y, n_select=6)[:6].tolist() == expected


def test_forward_selection_default_chooses_at_most_32_latents_greedily() -> None:
    X, y = _sparse_activations(n_samples=400, n_latents=80, seed=3)
    order = get_sorted_indices(X, y, "forward", k_max=64)
    assert sorted(order.tolist()) == list(range(80))
    greedy = order[:32].tolist()
    assert greedy == forward_selection_order(X, y, n_select=32)[:32].tolist()
    # past the greedy steps, latents follow in order of correlation with the label
    by_correlation = get_sorted_indices(X, y, "correlation").tolist()
    assert order[32:].tolist() == [i for i in by_correlation if i not in set(greedy)]
    # asking for more latents does not change the order
    assert get_sorted_indices(X, y, "forward", k_max=512).tolist() == order.tolist()
    assert get_sorted_indices(X, y, "forward").tolist() == order.tolist()


def test_get_selection_scores_for_forward_reproduce_its_order() -> None:
    X, y = _sparse_activations()
    scores = get_selection_scores(X, y, "forward", k_max=8)
    order = torch.argsort(scores, descending=True).tolist()
    assert order == forward_selection_order(X, y, n_select=8).tolist()


def test_get_save_metrics_path_keeps_default_names_and_separates_other_methods(
    tmp_path,
) -> None:
    common = dict(
        dataset="ds",
        hook_name="hook",
        reg_type="l1",
        model_name="gpt2",
        sae_results_path=tmp_path,
    )
    default = get_save_metrics_path(**common)  # type: ignore[arg-type]
    mean_diff = get_save_metrics_path(**common, selection_method="mean_diff")  # type: ignore[arg-type]
    auc = get_save_metrics_path(**common, selection_method="auc")  # type: ignore[arg-type]
    auc_binarized = get_save_metrics_path(
        **common,  # type: ignore[arg-type]
        selection_method="auc",
        binarize=True,
    )
    accuracy = get_save_metrics_path(**common, selection_method="accuracy")  # type: ignore[arg-type]

    assert default == mean_diff
    assert default.name == "ds_hook_l1.json"
    assert auc.name == "ds_hook_l1_sel-auc.json"
    assert auc_binarized.name == "ds_hook_l1_binarized_sel-auc.json"
    assert accuracy.name == "ds_hook_l1_sel-accuracy.json"
    assert len({default, auc, auc_binarized, accuracy}) == 4
    custom = get_save_metrics_path(
        **common,  # type: ignore[arg-type]
        selection_method=mean_diff_selection_fn(mean_act_normalization),
    )
    assert custom.name == "ds_hook_l1_sel-custom_mean_diff.json"
