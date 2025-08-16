# SAE Probes Benchmark

[![PyPI](https://img.shields.io/pypi/v/sae-probes?color=blue)](https://pypi.org/project/sae-probes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/sae-probes/sae-probes/actions/workflows/ci.yaml/badge.svg)](https://github.com/sae-probes/sae-probes/actions/workflows/ci.yaml)

This repository contains the code for the paper [_Are Sparse Autoencoders Useful? A Case Study in Sparse Probing_](https://arxiv.org/pdf/2502.16681), but has been reformatted into a Python package that will work with any SAE that can be loaded in [SAELens](https://github.com/jbloomAus/SAELens). This makes it easy to use the sparse probing tasks from the paper as a standalone SAE benchmark.

# Installation

```
pip install sae-probes
```

## Running evaluations

You can run benchmarks directly; any missing model activations are generated on demand. If you don't pass a `model_cache_path`, a temporary directory is used and cleaned up when the function completes. To persist activations across runs (recommended for repeated experiments), provide a `model_cache_path`.

## Training Probes

Probes can be trained directly on the model activations (baselines) or on SAE activations. In both cases, the following test data-balance settings are available: `"normal"`, `"scarcity"`, and `"imbalance"`. For more details about these settings, see the original paper. For the most standard sparse-probing benchmark, use the `normal` setting.

### SAE Probes

The most standard use of this library is as a sparse probing benchmark for SAEs using the `normal` setting. This is demonstrated below:

```python
from sae_probes import run_sae_evals
from sae_lens import SAE

# run the benchmark on a Gemma Scope SAE
release = "gemma-scope-2b-pt-res-canonical"
sae_id = "layer_12/width_16k/canonical"
sae = SAE.from_pretrained(release, sae_id)

run_sae_evals(
  sae=sae,
  model_name="gemma-2-2b",
  hook_name="blocks.12.hook_resid_post",
  reg_type="l1",
  setting="normal",
  results_path="/results/output/path",
  # model_cache_path is optional; if omitted, a temp dir is used and cleared after
  model_cache_path="/path/to/saved/activations",
  ks=[1, 16],
)
```

The sparse probing results for each dataset will be saved to `results_path` as a JSON file per dataset.

### Baseline Probes

You can now run baseline probes using a unified API that matches the SAE evaluation interface:

```python
from sae_probes import run_baseline_evals

# Run baseline probes with consistent API
run_baseline_evals(
  model_name="gemma-2-2b",
  hook_name="blocks.12.hook_resid_post",
  setting="normal",  # or "scarcity", "imbalance"
  results_path="/results/output/path",
  # model_cache_path is optional; if omitted, a temp dir is used and cleared after
  model_cache_path="/path/to/saved/activations",
)
```

#### Output Format

Both SAE and baseline probes now save results as **JSON files** with consistent structure:

- **SAE results**: `sae_probes_{model_name}/{setting}_setting/{dataset}_{hook_name}_{reg_type}.json`
- **Baseline results**: `baseline_results_{model_name}/{setting}_setting/{dataset}_{hook_name}_{method}.json`

Each JSON file contains a list with metrics and metadata for easy comparison between SAE and baseline approaches.

#### Optional: Pre-generating model activations

Pre-generating can speed up repeated runs and lets you inspect the saved tensors. It's optional because benchmarks will auto-generate missing activations on their first run if missing.

```python
from sae_probes import generate_dataset_activations

generate_dataset_activations(
  model_name="gemma-2-2b", # the TransformerLens name of the model
  hook_names=["blocks.12.hook_resid_post"], # Any TLens hook names
  batch_size=64,
  device="cuda",
  model_cache_path="/path/to/save/activations",
)
```

If you skip pre-generation, the benchmarks will create any missing activations automatically. Passing a `model_cache_path` persists them; if omitted, activations will be written to a temporary directory that is deleted after the run.

## Citation

If you use this code in your research, please cite:

```
@inproceedings{kantamnenisparse,
  title={Are Sparse Autoencoders Useful? A Case Study in Sparse Probing},
  author={Kantamneni, Subhash and Engels, Joshua and Rajamanoharan, Senthooran and Tegmark, Max and Nanda, Neel},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
