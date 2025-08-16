# %%
import glob
import os
import pickle

import pandas as pd
from tqdm import tqdm


# %%
def process_metrics(file, model_name):
    with open(file, "rb") as f:
        try:
            metrics = pickle.load(f)
            if model_name == "gemma-2-2b":
                for metric in metrics:
                    sae_id = metric["sae_id"]
                    name = "_".join(sae_id[2].split("/")[0].split("_")[1:])
                    l0 = sae_id[3]
                    rounded_l0 = round(float(l0))
                    metric["sae_id"] = f"{name}"
                    metric["sae_l0"] = rounded_l0
            return metrics
        except Exception:
            return None


def process_files(files, model_name):
    all_metrics = []
    bad_files = []

    file_iterator = tqdm(files)

    for file in file_iterator:
        metrics = process_metrics(file, model_name)
        if metrics:
            all_metrics.append(metrics)
        else:
            bad_files.append(file)

    return all_metrics, bad_files


def extract_sae_features(df, model_name):
    if model_name == "gemma-2-9b":
        df.loc[:, "sae_width"] = df["sae_id"].apply(
            lambda x: x.split("/")[1].split("_")[1]
        )
        df.loc[:, "sae_l0"] = df["sae_id"].apply(
            lambda x: int(x.split("/")[2].split("_")[2])
        )
    return df


def process_setting(setting, model_name):
    print(f"Processing {setting} setting for {model_name}...")

    # Create output directory
    output_dir = f"results/sae_probes_{model_name}/{setting}_setting"
    os.makedirs(output_dir, exist_ok=True)

    # Get file pattern based on setting
    file_pattern = f"data/sae_probes_{model_name}/{setting}_setting/*.pkl"

    # Process files
    files = glob.glob(file_pattern)
    print(file_pattern)
    print(len(files))
    all_metrics, bad_files = process_files(files, model_name)
    assert len(bad_files) == 0, f"Found {len(bad_files)} bad files in {setting} setting"

    # Create dataframe
    df = pd.DataFrame([item for sublist in all_metrics for item in sublist])

    # Save to CSV
    df.to_csv(f"{output_dir}/all_metrics.csv", index=False)

    # Print dataset length
    print(f"Total records in {setting} setting: {len(df)}")

    return df


# %%

for setting in ["normal", "scarcity", "class_imbalance", "label_noise"]:
    for model_name in ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]:
        process_setting(setting, model_name)
# %%
