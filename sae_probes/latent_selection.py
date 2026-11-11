"""Scores for choosing which SAE latents a k-sparse probe is trained on.

Every scorer takes the training activations ``X`` (n_samples x n_latents) and binary
labels ``y`` and returns one non-negative score per latent, where larger means the
latent separates the two classes better on the training set. The k latents with the
largest scores are used for the k-sparse probe.

- ``mean_diff``: the absolute difference between the class means of each latent's
  activation, after dividing the latent by its mean nonzero activation. This is the
  method from the original paper. It is sensitive to the magnitude of activations: a
  latent that fires on both classes but fires strongly on a few examples of one class
  can outrank a latent that fires on only one class.
- ``raw_mean_diff``: the same without the normalization, so latents with larger
  activations are preferred.
- ``correlation``: the absolute Pearson correlation between the latent's activation
  and the label. This is the mean difference divided by the latent's standard
  deviation, and gives the same ranking as Cohen's d. A latent whose class difference
  comes from a few very large activations has a large standard deviation, so it is
  ranked lower than by ``mean_diff``.
- ``firing_rate``: the absolute difference between the fraction of each class on
  which the latent is active, ignoring how strongly it fires.
- ``auc``: the area under the ROC curve of the latent's activation as a classifier
  score. It depends only on the ranking of activations, so it is unaffected by any
  monotonic rescaling of a latent's activations.
- ``accuracy``: the best training accuracy of a single threshold on the latent's
  activation. A 1-sparse probe is a single threshold on one latent, so for k=1 this is
  the selection rule that directly optimizes training accuracy.

All scorers are symmetric in the direction of the association (a latent that is active
on class 0 scores the same as an otherwise identical latent that is active on class 1).
"""

from typing import Callable, Literal

import torch

SelectionMethod = Literal[
    "mean_diff", "raw_mean_diff", "correlation", "firing_rate", "auc", "accuracy"
]
SelectionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SELECTION_METHODS: tuple[SelectionMethod, ...] = (
    "mean_diff",
    "raw_mean_diff",
    "correlation",
    "firing_rate",
    "auc",
    "accuracy",
)

# number of latents sorted at once by the rank-based scorers
_CHUNK_SIZE = 4096


def mean_act_normalization(X_sae: torch.Tensor) -> torch.Tensor:
    # take abs to handle negative values in the SAE activations for novel architectures.
    col_sums = X_sae.abs().sum(dim=0)
    col_nonzero_counts = (X_sae != 0).sum(dim=0)
    col_means = col_sums / (col_nonzero_counts + 1e-6)
    return X_sae / (col_means + 1e-6)


def mean_diff_scores(
    X: torch.Tensor,
    y: torch.Tensor,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """|mean of class 1 - mean of class 0| of the (optionally normalized) activations."""
    X = _as_float(X)
    X_normalized = normalize_fn(X) if normalize_fn else X
    diff = X_normalized[y == 1].mean(dim=0) - X_normalized[y == 0].mean(dim=0)
    return diff.abs()


def correlation_scores(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """|Pearson correlation| between each latent's activation and the label."""
    X = _as_float(X)
    label = (y == 1).to(X.dtype).to(X.device)
    label_centered = label - label.mean()
    covariance = (X * label_centered.unsqueeze(1)).mean(dim=0)
    denominator = X.std(dim=0, unbiased=False) * label.std(unbiased=False)
    scores = covariance.abs() / (denominator + 1e-12)
    # constant latents (or a single class) carry no information
    return torch.where(denominator > 0, scores, torch.zeros_like(scores))


def firing_rate_scores(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """|fraction of class 1 on which the latent is active - the same for class 0|."""
    return mean_diff_scores((X != 0).to(torch.float32), y)


def auc_scores(
    X: torch.Tensor, y: torch.Tensor, chunk_size: int = _CHUNK_SIZE
) -> torch.Tensor:
    """
    max(AUC, 1 - AUC) of each latent's activation as a classifier score, with tied
    activations given their average rank (so inactive latents and binarized
    activations are handled exactly).
    """
    X = _as_float(X)
    pos = (y == 1).to(X.device)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return torch.full((X.shape[1],), 0.5, dtype=X.dtype, device=X.device)
    scores = torch.empty(X.shape[1], dtype=X.dtype, device=X.device)
    for start in range(0, X.shape[1], chunk_size):
        values, order = X[:, start : start + chunk_size].sort(dim=0)
        ranks = _average_ranks_of_sorted(values)
        pos_sorted = pos[order]
        rank_sum = (ranks * pos_sorted).sum(dim=0)
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        scores[start : start + chunk_size] = torch.maximum(auc, 1 - auc).to(X.dtype)
    return scores


def accuracy_scores(
    X: torch.Tensor, y: torch.Tensor, chunk_size: int = _CHUNK_SIZE
) -> torch.Tensor:
    """
    The best training accuracy of a single threshold on each latent's activation,
    predicting either class above the threshold.
    """
    X = _as_float(X)
    pos = (y == 1).to(X.device)
    n = pos.numel()
    n_pos = int(pos.sum())
    scores = torch.empty(X.shape[1], dtype=X.dtype, device=X.device)
    for start in range(0, X.shape[1], chunk_size):
        values, order = X[:, start : start + chunk_size].sort(dim=0)
        pos_sorted = pos[order]
        # number of positives / negatives with an activation at or below each sorted value
        pos_below = pos_sorted.cumsum(dim=0)
        neg_below = (~pos_sorted).cumsum(dim=0)
        # predict class 1 above the threshold, class 0 at or below it
        correct = (n_pos - pos_below) + neg_below
        correct = torch.maximum(correct, n - correct)
        # a threshold can only sit between two distinct activation values
        is_boundary = torch.ones_like(pos_sorted)
        is_boundary[:-1] = values[1:] != values[:-1]
        correct = torch.where(is_boundary, correct, torch.zeros_like(correct))
        scores[start : start + chunk_size] = (correct.max(dim=0).values / n).to(X.dtype)
    return scores


def get_selection_scores(
    X: torch.Tensor,
    y: torch.Tensor,
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
) -> torch.Tensor:
    """
    One score per latent; larger scores are selected first. A callable
    `selection_method` is called with (X, y) and must return one score per latent.
    """
    if callable(selection_method):
        return selection_method(X, y)
    if selection_method == "mean_diff":
        return mean_diff_scores(X, y, mean_act_normalization)
    if selection_method == "raw_mean_diff":
        return mean_diff_scores(X, y)
    if selection_method == "correlation":
        return correlation_scores(X, y)
    if selection_method == "firing_rate":
        return firing_rate_scores(X, y)
    if selection_method == "auc":
        return auc_scores(X, y)
    if selection_method == "accuracy":
        return accuracy_scores(X, y)
    raise ValueError(f"Invalid selection_method: {selection_method}")


def get_sorted_indices(
    X_train_sae: torch.Tensor,
    y_train: torch.Tensor,
    selection_method: SelectionMethod | SelectionFn = "mean_diff",
) -> torch.Tensor:
    """Latent indices from best to worst by `selection_method` on the training set."""
    scores = get_selection_scores(X_train_sae, y_train, selection_method)
    return torch.argsort(scores, descending=True, stable=True)


def selection_method_name(selection_method: SelectionMethod | SelectionFn) -> str:
    if callable(selection_method):
        return getattr(selection_method, "__name__", "custom")
    return selection_method


def mean_diff_selection_fn(
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
) -> SelectionFn:
    """A selection function that ranks latents by the mean difference after `normalize_fn`."""

    def custom_mean_diff(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mean_diff_scores(X, y, normalize_fn)

    return custom_mean_diff


def _as_float(X: torch.Tensor) -> torch.Tensor:
    return X if X.is_floating_point() else X.to(torch.float32)


def _average_ranks_of_sorted(values: torch.Tensor) -> torch.Tensor:
    """
    1-based ranks of values that are sorted ascending along dim 0, where tied values
    all get the average of the ranks they span.
    """
    n = values.shape[0]
    index = torch.arange(n, device=values.device).unsqueeze(1).expand_as(values)
    starts_group = torch.ones_like(values, dtype=torch.bool)
    starts_group[1:] = values[1:] != values[:-1]
    ends_group = torch.ones_like(starts_group)
    ends_group[:-1] = starts_group[1:]
    first = torch.where(starts_group, index, torch.zeros_like(index)).cummax(0).values
    last = (
        torch.where(ends_group, index, torch.full_like(index, n - 1))
        .flip(0)
        .cummin(0)
        .values.flip(0)
    )
    return (first + last).to(torch.float64) / 2 + 1
