import glob
import os
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from sae_probes.constants import DATA_PATH
from sae_probes.generate_model_activations import ensure_dataset_activations


# DATA UTILS
@cache
def get_binary_df() -> pd.DataFrame:
    # returns a list of the data tags for all binary classification datasets
    df = pd.read_csv(
        DATA_PATH.parent.parent / "raw_data" / "probing_datasets_MASTER.csv"
    )
    # Filter for Binary Classification datasets
    binary_datasets = df[df["Data type"] == "Binary Classification"]
    return binary_datasets  # type: ignore


def get_numbered_binary_tags() -> list[str]:
    df = get_binary_df()
    return [name.split("/")[-1].split(".")[0] for name in df["Dataset save name"]]


@cache
def read_dataset_df(dataset_tag):
    # returns dataframe of df
    df = get_binary_df()
    # Find the dataset save name for the given dataset tag
    dataset_save_name = df[df["Dataset Tag"] == dataset_tag]["Dataset save name"].iloc[  # type: ignore
        0
    ]
    zipped_dataset_save_name = dataset_save_name + ".zst"
    # Read and return the dataset from the save location
    return pd.read_csv(DATA_PATH / zipped_dataset_save_name, compression="zstd")


def read_numbered_dataset_df(numbered_dataset_tag: str):
    # expects the form {number}_{dataset_tag}
    # Extract dataset tag from numbered format (e.g. "1_dataset" -> "dataset")
    dataset_tag = "_".join(numbered_dataset_tag.split("_")[1:])
    return read_dataset_df(dataset_tag)


def get_yvals(numbered_dataset_tag: str) -> np.ndarray:
    df = read_numbered_dataset_df(numbered_dataset_tag)
    le = LabelEncoder()
    yvals: np.ndarray = le.fit_transform(df["target"].values)  # type: ignore
    return yvals


def get_xvals(
    numbered_dataset_tag: str,
    hook_name: str,
    model_name: str,
    model_cache_path: str | Path,
):
    # Ensure activations exist for this dataset/hook
    ensure_dataset_activations(
        model_name=model_name,
        dataset_short_names=[numbered_dataset_tag],
        hook_names=[hook_name],
        model_cache_path=model_cache_path,
        device="cpu",
    )
    fname = (
        Path(model_cache_path)
        / f"model_activations_{model_name}/{numbered_dataset_tag}_{hook_name}.pt"
    )
    activations = torch.load(fname, weights_only=False)
    return activations


def get_xyvals(
    numbered_dataset_tag: str,
    hook_name: str,
    model_name: str,
    model_cache_path: str | Path,
    MAX_AMT: int = 1500,
):
    xvals = get_xvals(numbered_dataset_tag, hook_name, model_name, model_cache_path)
    yvals = get_yvals(numbered_dataset_tag)
    # Return only up to MAX_AMT samples
    xvals = xvals[:MAX_AMT]
    yvals = yvals[:MAX_AMT]  # type: ignore
    return xvals, yvals


def get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Split positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Calculate sizes ensuring they sum to num_train
    pos_train_size = int(np.ceil(pos_ratio * num_train))
    neg_train_size = num_train - pos_train_size

    # Same for test
    pos_test_size = int(np.ceil(pos_ratio * num_test))
    neg_test_size = num_test - pos_test_size

    # Sample train indices
    train_pos = np.random.choice(pos_indices, size=pos_train_size, replace=False)
    train_neg = np.random.choice(neg_indices, size=neg_train_size, replace=False)

    # Get remaining indices for test set
    remaining_pos = np.setdiff1d(pos_indices, train_pos)
    remaining_neg = np.setdiff1d(neg_indices, train_neg)

    # Sample test indices
    test_pos = np.random.choice(remaining_pos, size=pos_test_size, replace=False)
    test_neg = np.random.choice(remaining_neg, size=neg_test_size, replace=False)

    # Combine and shuffle indices
    train_indices = np.random.permutation(np.concatenate([train_pos, train_neg]))
    test_indices = np.random.permutation(np.concatenate([test_pos, test_neg]))

    return train_indices, test_indices


def get_xy_traintest_specify(
    num_train: int,
    numbered_dataset_tag: str,
    hook_name: str,
    model_name: str,
    model_cache_path: str | Path,
    pos_ratio: float = 0.5,
    MAX_AMT: int = 5000,
    seed: int = 42,
    num_test: int | None = None,
):
    X, y = get_xyvals(
        numbered_dataset_tag,
        hook_name,
        model_name,
        MAX_AMT=MAX_AMT,
        model_cache_path=model_cache_path,
    )
    if num_test is None:
        num_test = X.shape[0] - num_train - 1
    # Check if requested samples exceed available data
    if num_train + min(100, num_test) > X.shape[0]:  # type: ignore
        raise ValueError(
            f"Requested {num_train + 100} total samples (train={num_train}, test={100}) but only {X.shape[0]} samples available in dataset {numbered_dataset_tag}"
        )

    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio, seed
    )

    # Split data
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]  # type: ignore


def get_xy_traintest(
    num_train: int,
    numbered_dataset_tag: str,
    hook_name: str,
    model_name: str,
    model_cache_path: str | Path,
    MAX_AMT: int = 5000,
    seed: int = 42,
):
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        numbered_dataset_tag,
        hook_name,
        model_name,
        model_cache_path=model_cache_path,
        pos_ratio=0.5,
        MAX_AMT=MAX_AMT,
        seed=seed,
    )
    return X_train, y_train, X_test, y_test


@cache
def get_dataset_sizes():
    """
    Returns a DataFrame containing the dataset tag and total number of samples for each binary dataset.
    Uses read_numbered_dataset_df to read each dataset.

    Returns:
        pd.DataFrame: DataFrame with columns ['numbered_dataset_tag', 'num_samples']
    """
    # Initialize lists to store results
    dataset_tags = get_numbered_binary_tags()
    dataset_sizes = {}
    # Process each dataset file
    for i, dataset_tag in enumerate(dataset_tags):
        df = read_numbered_dataset_df(dataset_tag)
        num_samples = len(df)
        dataset_sizes[dataset_tag] = num_samples
    return dataset_sizes


def get_training_sizes():
    min_size, max_size, num_points = 1, 10, 20
    points = np.unique(
        np.round(np.logspace(min_size, max_size, num=num_points, base=2)).astype(int)
    )
    return points.tolist()


def get_class_imbalance():
    min_size, max_size, num_points = 0.05, 0.95, 19
    points = np.linspace(min_size, max_size, num=num_points)
    return points.tolist()


def get_classimabalance_num_train(
    numbered_dataset: str, min_num_test: int = 100
) -> tuple[int, int]:
    # gives the maximum number of num_train possible
    y: np.ndarray = get_yvals(numbered_dataset)
    points = get_class_imbalance()
    min_p, max_p = min(points), max(points)
    num_pos = np.sum(y)
    num_neg = len(y) - num_pos
    max_total_neg = num_neg / (1 - min_p)
    max_total_pos = num_pos / max_p
    max_total = int(min(max_total_neg, max_total_pos))
    num_train = min(max_total - min_num_test, 1024)  # 100 is min number test
    num_test = max(100, max_total - num_train - 1)
    return num_train, num_test


def corrupt_ytrain(ytrain, frac):
    assert 0 <= frac <= 0.5
    np.random.seed(42)
    # Get indices to flip
    num_to_flip = int(len(ytrain) * frac)
    flip_indices = np.random.choice(len(ytrain), size=num_to_flip, replace=False)

    # Create copy and flip selected labels
    ytrain_corrupted = ytrain.copy()
    ytrain_corrupted[flip_indices] = 1 - ytrain_corrupted[flip_indices]

    return ytrain_corrupted


def get_corrupt_frac():
    min_size, max_size, num_points = 0, 0.5, 11
    points = np.linspace(min_size, max_size, num=num_points)
    return points.tolist()


def get_OOD_datasets(translation: bool = True) -> list[str]:
    # translation tells us if we want that one living room dataset
    dataset_names = glob.glob(str(DATA_PATH / "OOD data" / "*.csv.zst"))
    # Extract dataset names from paths by taking the base filename without _OOD.csv.zst
    if translation:
        datasets = [
            os.path.basename(path).replace("_OOD.csv.zst", "") for path in dataset_names
        ]
    else:
        datasets = [
            os.path.basename(path).replace("_OOD.csv.zst", "")
            for path in dataset_names
            if "translation" not in path
        ]
    return [d.replace(".csv.zst", "") for d in datasets]


def get_xy_OOD(
    dataset: str,
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
):
    ensure_dataset_activations(
        model_name=model_name,
        dataset_short_names=[dataset],
        hook_names=[hook_name],
        model_cache_path=model_cache_path,
        device="cpu",
        OOD=True,
    )
    X = torch.load(
        Path(model_cache_path)
        / f"model_activations_{model_name}_OOD/{dataset}_OOD_{hook_name}.pt",
        weights_only=False,
    )
    df = pd.read_csv(DATA_PATH / f"OOD data/{dataset}_OOD.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["target"].values)  # type: ignore
    return X, y


def get_OOD_traintest(
    dataset: str,
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
):
    X_train, y_train, _, _ = get_xy_traintest_specify(
        num_train=1024,
        numbered_dataset_tag=dataset,
        hook_name=hook_name,
        model_name=model_name,
        MAX_AMT=1500,
        pos_ratio=0.5,
        num_test=0,
        model_cache_path=model_cache_path,
    )
    X_test, y_test = get_xy_OOD(dataset, model_name, hook_name, model_cache_path)
    return X_train, y_train, X_test, y_test


def get_datasets(
    model_name: str,
    hook_name: str,
    model_cache_path: str | Path,
):
    dataset_sizes = get_dataset_sizes()
    directory = Path(model_cache_path) / f"model_activations_{model_name}"
    files = os.listdir(str(directory))
    suffix = f"_{hook_name}.pt"
    datasets: set[str] = set()
    for file in files:
        if file.endswith(suffix):
            dataset = file[: -len(suffix)]
            if dataset in dataset_sizes.keys():
                datasets.add(dataset)
    return sorted(list(datasets))
