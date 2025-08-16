# %%
import json
from collections import defaultdict

import torch
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    MODEL_CONFIGS,
    get_all_hf_repo_autoencoders,
    load_dictionary_learning_sae,
)
from sae_bench.sae_bench_utils import general_utils


def get_gemma_2_2b_sae_ids(layer):
    assert layer == 12

    repo_ids = [
        "canrager/saebench_gemma-2-2b_width-2pow14_date-0107",
        "adamkarvonen/temp",
    ]
    model_name = "gemma-2-2b"
    exclude_keywords = [
        "checkpoints",
        "eval_results.json",
        "random_seed_eval_results",
        "random_seeds_google_gemma-2-2b_top_k",
        "topk-finetune",
        "finetune",
    ]
    include_keywords = []

    all_sae_locations = []
    for repo_id in repo_ids:
        print(f"\n\n\nEvaluating {model_name} with {repo_id}\n\n\n")

        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]

        torch_dtype = torch.float32

        sae_locations = get_all_hf_repo_autoencoders(repo_id)

        sae_locations = general_utils.filter_keywords(
            sae_locations,
            exclude_keywords=exclude_keywords,
            include_keywords=include_keywords,
        )

        cfg_paths = [
            f"downloaded_saes/{repo_id.replace('/', '_')}/{loc}/eval_results.json"
            for loc in sae_locations
        ]
        l0s = [json.load(open(cfg_path))["l0"] for cfg_path in cfg_paths]
        all_sae_locations.extend(
            [(repo_id, model_name, loc, l0) for loc, l0 in zip(sae_locations, l0s)]
        )

    return all_sae_locations


def get_gemma_2_2b_sae_ids_largest_l0(layer):
    assert layer == 12

    all_sae_locations = get_gemma_2_2b_sae_ids(layer)

    # Group SAE locations by their base path (everything except the last part)
    base_paths = defaultdict(list)
    for repo_id, model_name, loc, l0 in all_sae_locations:
        # Split path and group by everything except last component
        path_parts = loc.split("/")
        base = "/".join(path_parts[:-1])
        base_paths[base].append((repo_id, model_name, loc, l0))

    filtered_locations = []
    for base, locations in base_paths.items():
        if len(locations) == 1:
            filtered_locations.extend(locations)
            continue

        target_l0 = 200
        best_distance_to_target_l0 = float("inf")
        best_loc = None
        for repo_id, model_name, loc, l0 in locations:
            distance_to_target_l0 = abs(l0 - target_l0)
            if distance_to_target_l0 < best_distance_to_target_l0:
                best_distance_to_target_l0 = distance_to_target_l0
                best_loc = (repo_id, model_name, loc)

        if best_loc:
            filtered_locations.append(best_loc)

    return filtered_locations


def load_gemma_2_2b_sae(sae_location):
    repo_id, model_name, sae_location, l0 = sae_location

    return load_dictionary_learning_sae(
        repo_id=repo_id,
        location=sae_location,
        layer=None,
        model_name=model_name,
        device="cpu",
        dtype=torch.float32,
    )
