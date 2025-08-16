# %%

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from tqdm import tqdm

torch.set_grad_enabled(False)

# %%
layer = 20
dataset = "110_aimade_humangpt3"
activations = torch.load(
    f"data/model_activations_gemma-2-9b/{dataset}_blocks.{layer}.hook_resid_post.pt"
)
device = "cuda:1"
# From manual examanition class 1 = AI, class 0 = Human
targets = pd.read_csv(f"data/cleaned_data/{dataset}.csv")["target"].tolist()
targets = torch.tensor(targets, device="cpu")
# %%

sae = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res",
    sae_id=f"layer_{layer}/width_131k/average_l0_114",
    device="cpu",
)[0].to(device)

# %%

batch_size = 100

sae_activations = []
for i in tqdm(range(0, len(activations), batch_size)):
    batch = activations[i : i + batch_size].to(device)
    sae_activations.append(sae.encode(batch).cpu())
sae_activations = torch.cat(sae_activations, dim=0)
# %%


text = pd.read_csv(f"data/cleaned_data/{dataset}.csv")
text = text["prompt"].tolist()

from transformers import AutoTokenizer

model_name = "google/gemma-2-9b"
device = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.truncation_side = "left"
tokenizer.padding_side = "right"


last_tokens = []
for i in tqdm(range(len(text))):
    last_tokens.append(
        tokenizer(text[i], return_tensors="pt").to(device)["input_ids"][:, -1]
    )

last_tokens = torch.cat(last_tokens, dim=0)

mean_diffs = sae_activations[targets == 1].mean(axis=0) - sae_activations[
    targets == 0
].mean(axis=0)

# %%
import matplotlib.pyplot as plt


def plot_feature_analysis(
    sae_activations,
    targets,
    last_tokens,
    tokenizer,
    top_k=3,
    cutoff=100,
    figsize=(3.25, 1.6),  # Made overall figure wider
    fontsize=6,
    ylabel_fontsize=None,
    label_rotation_left=0,
    label_rotation_right=0,
    bar_alpha=0.7,
    ytick_fontsize=4.5,
):
    """
    Plot analysis of top features and token distributions for AI vs Human text.

    Args:
        sae_activations: Tensor of SAE activations
        targets: Tensor of binary targets (0=Human, 1=AI)
        last_tokens: Tensor of last tokens for each text
        tokenizer: Tokenizer for decoding tokens
        top_k: Number of top features to show
        cutoff: Minimum frequency for tokens to be included
        figsize: Figure size as (width, height)
        fontsize: Base font size for labels
        ylabel_fontsize: Font size for y-axis labels (defaults to fontsize if None)
        label_rotation_left: Rotation angle for x-tick labels in left plot
        label_rotation_right: Rotation angle for x-tick labels in right plot
        bar_alpha: Alpha/transparency for bars
        ytick_fontsize: Font size for y-axis numerical labels
    """
    if ylabel_fontsize is None:
        ylabel_fontsize = fontsize

    # Calculate mean differences between AI and Human activations
    mean_diffs = sae_activations[targets == 1].mean(axis=0) - sae_activations[
        targets == 0
    ].mean(axis=0)

    # Get top features by difference magnitude
    top_by_diff = torch.flip(torch.argsort(torch.abs(mean_diffs))[-top_k:], dims=[0])
    top_diffs = mean_diffs[top_by_diff].abs()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.1, 1]}
    )  # Made left plot wider
    # Plot 1: Top Features
    explanations = ["periods", "spaces", "spaces"]
    labels = []
    for idx, exp in zip(top_by_diff, explanations):
        labels.append(f"{idx.item()}:\n{exp}")

    ax1.bar(
        range(len(top_diffs)), top_diffs.cpu().numpy(), alpha=bar_alpha, color="green"
    )
    ax1.set_xlabel("Latent", fontsize=fontsize)
    ax1.set_ylabel("|Mean Act Difference|\n(AI - Human)", fontsize=ylabel_fontsize)
    ax1.set_xticks(range(len(top_diffs)))
    ax1.set_xticklabels(labels, rotation=label_rotation_left, fontsize=ytick_fontsize)
    ax1.tick_params(axis="y", labelsize=ytick_fontsize)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Token Distribution
    # Count token frequencies
    human_token_counts = {}
    ai_token_counts = {}
    for i, token in enumerate(last_tokens):
        token = token.item()
        if targets[i] == 0:
            human_token_counts[token] = human_token_counts.get(token, 0) + 1
        else:
            ai_token_counts[token] = ai_token_counts.get(token, 0) + 1

    # Get common tokens above cutoff
    common_tokens = {
        t
        for t in set(human_token_counts) | set(ai_token_counts)
        if max(human_token_counts.get(t, 0), ai_token_counts.get(t, 0)) >= cutoff
    }

    # Sort by total frequency
    common_tokens = sorted(
        common_tokens,
        key=lambda t: human_token_counts.get(t, 0) + ai_token_counts.get(t, 0),
        reverse=True,
    )

    # Prepare data for plotting
    human_freqs = [human_token_counts.get(t, 0) for t in common_tokens]
    ai_freqs = [ai_token_counts.get(t, 0) for t in common_tokens]

    # Create readable token labels with wrapping for long labels
    token_labels = ["period", "quest.\nmark", "space", "excl.\npoint"]

    # Plot grouped bars
    x = np.arange(len(common_tokens))
    width = 0.35

    ax2.bar(
        x - width / 2, human_freqs, width, label="Human", alpha=bar_alpha, color="blue"
    )
    ax2.bar(x + width / 2, ai_freqs, width, label="AI", alpha=bar_alpha, color="red")

    ax2.set_xlabel("Last Token", fontsize=fontsize)
    ax2.set_ylabel("Total Activation", fontsize=ylabel_fontsize)
    ax2.set_xticks(x)

    ax2.set_xticklabels(
        token_labels, rotation=label_rotation_right, fontsize=ytick_fontsize
    )
    ax2.tick_params(axis="y", labelsize=ytick_fontsize)
    ax2.legend(
        fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.1, 1.2), framealpha=0.9
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Example usage:
fig = plot_feature_analysis(
    sae_activations=sae_activations,
    targets=targets,
    last_tokens=last_tokens,
    tokenizer=tokenizer,
    label_rotation_left=0,  # 45 degree rotation for left plot
    label_rotation_right=0,  # 45 degree rotation for right plot
)

# Show and save the figure
plt.show()
fig.savefig("plots/ai_vs_human_analysis.pdf", bbox_inches="tight")
plt.close()
# %%
