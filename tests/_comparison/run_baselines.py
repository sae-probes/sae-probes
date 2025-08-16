import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils_data import (
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_datasets,
    get_disagree_glue,
    get_glue_traintest,
    get_layers,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_OOD_traintest,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from .utils_sae import get_xy_glue_sae, get_xy_OOD_sae
from .utils_training import (
    find_best_knn,
    find_best_mlp,
    find_best_pcareg,
    find_best_reg,
    find_best_xgboost,
)

dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()
methods = {
    "logreg": find_best_reg,
    "pca": find_best_pcareg,
    "knn": find_best_knn,
    "xgboost": find_best_xgboost,
    "mlp": find_best_mlp,
}


"""
FUNCTIONS FOR STANDARD CONDITIONS 
"""


def run_baseline_dataset_layer(
    layer, numbered_dataset, method_name, model_name="gemma-2-9b"
):
    savepath = f"data/baseline_results_{model_name}/normal/allruns/layer{layer}_{numbered_dataset}_{method_name}.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )

    # Run method and get metrics
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)

    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name}
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_normal(model_name="gemma-2-9b"):
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    for method_name in methods.keys():
        for layer in get_layers(model_name):
            for dataset in shuffled_datasets:
                # print(layer, dataset, method_name)
                val = run_baseline_dataset_layer(
                    layer, dataset, method_name, model_name=model_name
                )
                # print(val)


def coalesce_all_baseline_normal(model_name="gemma-2-9b"):
    # takes individual csvs and makes it into one big csv
    i = 0
    for layer in get_layers(model_name):
        all_results = []
        for dataset in datasets:
            for method_name in methods.keys():
                savepath = f"data/baseline_results_{model_name}/normal/allruns/layer{layer}_{dataset}_{method_name}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    all_results.append(df)
                    i += 1
                else:
                    print(f"Missing file {layer}, {method_name}, {dataset}")
                    # raise ValueError(f'Missing file {layer}, {method_name}, {dataset}')

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            layer_savepath = f"results/baseline_probes_{model_name}/normal_settings/layer{layer}_results.csv"
            os.makedirs(os.path.dirname(layer_savepath), exist_ok=True)
            combined_df.to_csv(layer_savepath, index=False)


"""
FUNCTIONS FOR DATA SCARCITY CONDITION
"""


def run_baseline_scarcity(
    num_train, numbered_dataset, method_name, model_name="gemma-2-9b", layer=20
):
    savepath = f"data/baseline_results_{model_name}/scarcity/allruns/layer{layer}_{numbered_dataset}_{method_name}_numtrain{num_train}.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
    if num_train > size - 100:
        # we dont have enough test examples
        return
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )
    # Run method and get metrics
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # print(metrics)
    # Create row with dataset and method metrics and save to csv
    row = {"dataset": numbered_dataset, "method": method_name, "num_train": num_train}
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_scarcity(model_name="gemma-2-9b", layer=20):
    assert layer in get_layers(model_name)
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    train_sizes = get_training_sizes()
    for method_name in methods.keys():
        for train in train_sizes:
            for dataset in shuffled_datasets:
                val = run_baseline_scarcity(
                    train, dataset, method_name, model_name=model_name, layer=layer
                )
                # print(method_name, train, dataset, val)


def coalesce_all_scarcity(model_name="gemma-2-9b", layer=20):
    # takes individual csvs and makes it into one big csv
    all_results = []
    train_sizes = get_training_sizes()

    # Create directories if they don't exist
    dataset_path = f"data/baseline_results_{model_name}/scarcity/by_dataset"
    allpath = f"results/baseline_probes_{model_name}/scarcity/"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)

    for dataset in datasets:
        dataset_results = []
        for num_train in train_sizes:
            for method_name in methods.keys():
                savepath = f"data/baseline_results_{model_name}/scarcity/allruns/layer{layer}_{dataset}_{method_name}_numtrain{num_train}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    if num_train + 100 <= dataset_sizes[dataset]:
                        raise ValueError(
                            f"Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})"
                        )
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = f"{dataset_path}/{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = f"{allpath}/all_results.csv"
        print(summary_savepath)
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR CLASS IMBALANCE CONDITION
"""


def run_baseline_class_imbalance(
    dataset_frac, numbered_dataset, method_name, model_name="gemma-2-9b", layer=20
):
    assert 0 < dataset_frac < 1
    dataset_frac = round(dataset_frac * 20) / 20
    savepath = f"data/baseline_results_{model_name}/imbalance/allruns/layer{layer}_{numbered_dataset}_{method_name}_frac{dataset_frac}.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    num_train, num_test = get_classimabalance_num_train(numbered_dataset)
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        numbered_dataset,
        layer,
        pos_ratio=dataset_frac,
        model_name=model_name,
        num_test=num_test,
    )
    # Run method and get metrics
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": dataset_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_class_imbalance(model_name="gemma-2-9b", layer=20):
    assert layer in get_layers(model_name)
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_class_imbalance()
    i = 0
    for method_name in methods.keys():
        for frac in fracs:
            for dataset in shuffled_datasets:
                val = run_baseline_class_imbalance(
                    frac, dataset, method_name, model_name=model_name, layer=layer
                )


def coalesce_all_imbalance(model_name="gemma-2-9b", layer=20):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = f"data/baseline_results_{model_name}/imbalance/by_dataset"
    allpath = f"results/baseline_probes_{model_name}/imbalance"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_class_imbalance()
    i = 0
    for dataset in datasets:
        dataset_results = []
        for frac in fracs:
            for method_name in methods.keys():
                frac = round(frac * 20) / 20
                savepath = f"data/baseline_results_{model_name}/imbalance/allruns/layer{layer}_{dataset}_{method_name}_frac{frac}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    i += 1
                    # raise ValueError(f'Missing file {savepath}, {dataset} ({frac}/{dataset_sizes[dataset]})')
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = f"{dataset_path}/{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = f"{allpath}/all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR CORRUPT CONDITIONS
"""


def run_baseline_corrupt(
    corrupt_frac, numbered_dataset, method_name, model_name="gemma-2-9b", layer=20
):
    assert 0 <= corrupt_frac <= 0.5
    corrupt_frac = round(corrupt_frac * 20) / 20
    savepath = f"data/baseline_results_{model_name}/corrupt/allruns/layer{layer}_{numbered_dataset}_{method_name}_corrupt{corrupt_frac}.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    size = dataset_sizes[numbered_dataset]
    num_train = min(size - 100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )
    y_train = corrupt_ytrain(y_train, corrupt_frac)
    # Run method and get metrics
    method = methods[method_name]
    metrics = method(X_train, y_train, X_test, y_test)
    # Create row with dataset and method metrics and save to csv
    row = {
        "dataset": numbered_dataset,
        "method": method_name,
        "ratio": corrupt_frac,
        "num_train": num_train,
    }
    for metric_name, metric_value in metrics.items():
        row[f"{metric_name}"] = metric_value
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_baseline_corrupt(model_name="gemma-2-9b", layer=20):
    assert layer in get_layers(model_name)
    shuffled_datasets = get_datasets().copy()
    np.random.shuffle(shuffled_datasets)
    fracs = get_corrupt_frac()
    for method_name in ["logreg"]:
        for frac in fracs:
            for dataset in shuffled_datasets:
                val = run_baseline_corrupt(
                    frac, dataset, method_name, model_name=model_name, layer=layer
                )
                # print(val, method_name, frac, dataset)


def coalesce_all_corrupt(model_name="gemma-2-9b", layer=20):
    # takes individual csvs and makes it into one big csv
    all_results = []
    # Create directories if they don't exist
    dataset_path = f"data/baseline_results_{model_name}/corrupt/by_dataset"
    allpath = f"results/baseline_probes_{model_name}/corrupt"
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(allpath, exist_ok=True)
    fracs = get_corrupt_frac()
    for dataset in datasets:
        dataset_results = []
        for frac in fracs:
            for method_name in ["logreg"]:
                frac = round(frac * 20) / 20
                savepath = f"data/baseline_results_{model_name}/corrupt/allruns/layer{layer}_{dataset}_{method_name}_corrupt{frac}.csv"
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    dataset_results.append(df)
                    all_results.append(df)
                else:
                    raise ValueError(
                        f"Missing file {method_name}, {dataset} ({frac}/{dataset_sizes[dataset]})"
                    )
                    # print(f'Missing file {method_name}, {dataset} ({num_train}/{dataset_sizes[dataset]})')

        # Save dataset-specific results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            dataset_savepath = f"{dataset_path}/{dataset}.csv"
            dataset_df.to_csv(dataset_savepath, index=False)

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_savepath = f"{allpath}/all_results.csv"
        combined_df.to_csv(summary_savepath, index=False)


"""
FUNCTIONS FOR OOD EXPERIMENTS. This is the only regime where the SAE and baseline runs are done together
"""


def run_datasets_OOD(model_name="gemma-2-9b", runsae=True, layer=20, translation=False):
    # runs the baseline and sae probes for OOD generalization
    # trains on normal data but tests on the OOD activations
    # run_sae should be true to run the sae generalization experiments
    # translation = True runs the probe on 66_living_room translated into different languages.
    # You can likely set this to False
    datasets = get_OOD_datasets(translation=translation)
    results = []

    for dataset in tqdm(datasets):
        X_train, y_train, X_test, y_test = get_OOD_traintest(
            dataset=dataset, model_name=model_name, layer=layer
        )
        metrics = find_best_reg(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, penalty="l2"
        )
        tosave = {"dataset": dataset, "test_auc_baseline": metrics["test_auc"]}
        if runsae:
            X_train_sae, y_train_sae, X_test_sae, y_test_sae = get_xy_OOD_sae(dataset)
            metrics_sae = find_best_reg(
                X_train=X_train_sae,
                y_train=y_train_sae,
                X_test=X_test_sae,
                y_test=y_test_sae,
                penalty="l1",
            )
            tosave["test_auc_sae"] = metrics_sae["test_auc"]
        results.append(tosave)

    # Create and save results dataframe
    os.makedirs(f"results/baseline_probes_{model_name}/ood/", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"results/baseline_probes_{model_name}/ood/all_results.csv", index=False
    )


def run_glue():
    # this function is used to do the GLUE CoLA investigation Section 4.3.1
    # all three runs train on normal GLUE CoLA dataset
    # run 0 tests on the original target labels (dirty)
    # run 1 tests on ensembled, clean labels from a council of LLMs
    # run 2 tests only on the examples where clean labels disagree with the original labels
    results = []
    for run in range(3):
        toget = "original_target" if run == 0 else "ensemble"

        # Baseline run
        X_train, y_train, X_test, y_test = get_glue_traintest(toget)
        if run == 2:
            indices = get_disagree_glue()
            X_test = X_test[indices]
            y_test = y_test[indices]
        metrics = find_best_reg(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            penalty="l2",
        )
        results.append(
            {
                "run_num": run,
                "run_type": "baseline",
                "test_auc": metrics["test_auc"],
                "test_acc": metrics["test_acc"],
            }
        )

        # SAE runs
        for k in [128, 1]:
            X_train, y_train, X_test, y_test = get_xy_glue_sae(toget=toget, k=k)
            if run == 2:
                indices = get_disagree_glue()
                X_test = X_test[indices]
                y_test = y_test[indices]
            metrics_sae = find_best_reg(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                penalty="l1",
            )
            results.append(
                {
                    "run_num": run,
                    "run_type": f"sae_{k}",
                    "test_auc": metrics_sae["test_auc"],
                    "test_acc": metrics_sae["test_acc"],
                }
            )

    results_df = pd.DataFrame(results)
    os.makedirs("results/investigate/", exist_ok=True)
    results_df.to_csv(
        "results/investigate/87_glue_cola_investigate_probes.csv", index=False
    )


def latent_performance(dataset="66_living-room"):
    # calculates the single latent performance as a k=1 classifier for the top 8 latents by mean difference
    results = []
    X_train, y_train, X_test, y_test, top_by_average_diff = get_xy_OOD_sae(
        dataset,
        k=8,
        model_name="gemma-2-9b",
        layer=20,
        return_indices=True,
        num_train=1500,
    )
    print(top_by_average_diff)
    metrics, classifier = find_best_reg(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        penalty="l1",
        return_classifier=True,
    )
    results.append(
        {
            "latent": "all",
            "ood_auc": metrics["test_auc"],
            "val_auc": metrics["val_auc"],
            "coef": "None",
        }
    )
    coefs = classifier.coef_[0]
    bar = tqdm(range(8))
    for k in bar:
        latent = top_by_average_diff[k].item()
        X_train_lat = X_train[:1024, k : k + 1]
        X_test_lat = X_test[:, k : k + 1]
        metrics = find_best_reg(
            X_train=X_train_lat,
            y_train=y_train,
            X_test=X_test_lat,
            y_test=y_test,
            penalty="l1",
        )
        results.append(
            {
                "latent": latent,
                "ood_auc": metrics["test_auc"],
                "val_auc": metrics["val_auc"],
                "coef": coefs[k],
            }
        )
        bar.set_postfix(results[-1])

    results_df = pd.DataFrame(results)
    os.makedirs(
        f"results/sae_probes_gemma-2-9b/OOD/OOD_latents/{dataset}", exist_ok=True
    )
    results_df.to_csv(
        f"results/sae_probes_gemma-2-9b/OOD/OOD_latents/{dataset}/{dataset}_latent_aucs_raw.csv",
        index=False,
    )
    # finds the performance of the top 1 latent for a given dataset


# run_datasets_OOD()


def ood_pruning(dataset="66_living-room"):
    # does OOD Pruning
    # We use o1 to rank the latents by usefulness to the task via auto-interp explanations,
    # and prune the least helpful latents to see if that helps performance
    # section
    fname = f"results/sae_probes_gemma-2-9b/OOD/OOD_latents/{dataset}/{dataset}_latent_aucs.csv"
    df = pd.read_csv(fname)
    df = df.sort_values("Relevance")
    X_train, y_train, X_test, y_test, top_by_average_diff = get_xy_OOD_sae(
        dataset,
        k=8,
        model_name="gemma-2-9b",
        layer=20,
        return_indices=True,
        num_train=1500,
    )

    results = []
    bar = tqdm(range(1, 9))
    for k in bar:
        # Get top k latents by relevance
        top_k_latents = df.head(k)["latent"].values

        # Find indices of these latents in top_by_average_diff
        indices = [
            i for i, x in enumerate(top_by_average_diff) if x.item() in top_k_latents
        ]

        # Index X_train and X_test with these indices
        X_train_filtered = X_train[:, indices]
        X_test_filtered = X_test[:, indices]

        # Run find_best_reg
        metrics = find_best_reg(
            X_train=X_train_filtered,
            y_train=y_train,
            X_test=X_test_filtered,
            y_test=y_test,
            penalty="l1",
        )
        results.append(
            {"k": k, "ood_auc": metrics["test_auc"], "val_auc": metrics["val_auc"]}
        )
        bar.set_postfix(results[-1])
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"results/sae_probes_gemma-2-9b/OOD/OOD_latents/{dataset}/{dataset}_pruned.csv",
        index=False,
    )


def examine_glue_classifier():
    # finds the prompts where baseline classifier most disagrees with
    # the given label. This allows us to see if baselines are able
    # to find incorrect labels as well. Table 7
    X_train, y_train, X_test, y_test_og = get_glue_traintest(toget="original_target")
    _, _, _, y_test_ens = get_glue_traintest(toget="ensemble")
    _, classifier = find_best_reg(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test_og,
        penalty="l2",
        return_classifier=True,
    )
    # Get predictions and probabilities
    probs = classifier.predict_proba(X_test)
    prob_class_0 = probs[:, 0]  # Probability of class 0

    # Find indices where actual is 1 but model is most confident it should be 0
    mask = y_test_og == 1
    confident_wrong_indices = np.argsort(prob_class_0[mask])[-5:][
        ::-1
    ]  # Top 5 most confident mistakes
    actual_indices = np.where(mask)[0][confident_wrong_indices]

    # Load prompts
    df = pd.read_csv("results/investigate/87_glue_cola_investigate.csv")
    prompts = df["prompt"].tolist()

    # Print table header
    print("\nMost confident mistakes (where true label is 1 but model predicts 0)")
    print("-" * 100)
    print(f"{'Prompt':<60} | {'P(y=0)':<10} | {'Original':<8} | {'Ensemble':<8}")
    print("-" * 100)

    # Print each row
    for idx in actual_indices:
        print(
            f"{prompts[idx]:<60} | {prob_class_0[idx]:.3f}     | {y_test_og[idx]:<8} | {y_test_ens[idx]:<8}"
        )


if __name__ == "__main__":
    """
    Note: we do not recommend you run the functions like this.
    Each run_all file can be run in parallel instances using a 
    bash script to considerably speed up the runs.
    """
    run_all_baseline_normal(
        "gemma-2-9b"
    )  # runs baseline probes in standard conditions on 4 evenly spaced layers
    coalesce_all_baseline_normal(model_name="gemma-2-9b")

    run_all_baseline_scarcity("gemma-2-9b", layer=20)
    coalesce_all_scarcity("gemma-2-9b", layer=20)

    run_all_baseline_class_imbalance("gemma-2-9b", layer=20)
    coalesce_all_imbalance("gemma-2-9b", layer=20)

    run_all_baseline_corrupt("gemma-2-9b", layer=20)
    coalesce_all_corrupt("gemma-2-9b", layer=20)

    run_datasets_OOD("gemma-2-9b", runsae=True, layer=20, translation=False)
