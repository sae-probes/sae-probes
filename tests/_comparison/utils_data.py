import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False


# DATA UTILS
def get_binary_df():
    # returns a list of the data tags for all binary classification datasets
    df = pd.read_csv("data/probing_datasets_MASTER.csv")
    # Filter for Binary Classification datasets
    binary_datasets = df[df["Data type"] == "Binary Classification"]
    return binary_datasets


def get_numbered_binary_tags():
    df = get_binary_df()
    return [name.split("/")[-1].split(".")[0] for name in df["Dataset save name"]]


def read_dataset_df(dataset_tag):
    # returns dataframe of df
    df = get_binary_df()
    # Find the dataset save name for the given dataset tag
    dataset_save_name = df[df["Dataset Tag"] == dataset_tag]["Dataset save name"].iloc[
        0
    ]
    # Read and return the dataset from the save location
    return pd.read_csv(f"data/{dataset_save_name}")


def read_numbered_dataset_df(numbered_dataset_tag):
    # expects hte form {number}_{dataset_tag}
    # Extract dataset tag from numbered format (e.g. "1_dataset" -> "dataset")
    dataset_tag = "_".join(numbered_dataset_tag.split("_")[1:])
    return read_dataset_df(dataset_tag)


def get_yvals(numbered_dataset_tag):
    df = read_numbered_dataset_df(numbered_dataset_tag)
    le = LabelEncoder()
    yvals = le.fit_transform(df["target"].values)
    return yvals


def get_xvals(numbered_dataset_tag, layer, model_name="gemma-2-9b"):
    # assert layer in [9,20,31,41,'embed'], 'invalid layer provided'
    if layer == "embed":
        fname = (
            f"data/model_activations_{model_name}/{numbered_dataset_tag}_hook_embed.pt"
        )
    else:
        fname = f"data/model_activations_{model_name}/{numbered_dataset_tag}_blocks.{layer}.hook_resid_post.pt"
    activations = torch.load(fname, weights_only=False)
    # print(f"Shape of {numbered_dataset_tag} at layer {layer}: {activations.shape}")
    return activations


def get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=1500):
    xvals = get_xvals(numbered_dataset_tag, layer, model_name)
    yvals = get_yvals(numbered_dataset_tag)
    # Return only up to MAX_AMT samples
    xvals = xvals[:MAX_AMT]
    yvals = yvals[:MAX_AMT]
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
    num_train,
    numbered_dataset_tag,
    layer,
    model_name,
    pos_ratio=0.5,
    MAX_AMT=5000,
    seed=42,
    num_test=None,
):
    X, y = get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=MAX_AMT)
    if num_test is None:
        num_test = X.shape[0] - num_train - 1
    # Check if requested samples exceed available data
    if num_train + min(100, num_test) > X.shape[0]:
        raise ValueError(
            f"Requested {num_train + 100} total samples (train={num_train}, test={100}) but only {X.shape[0]} samples available in dataset {numbered_dataset_tag}"
        )

    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio, seed
    )

    # Split data
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def get_xy_traintest(
    num_train, numbered_dataset_tag, layer, model_name, MAX_AMT=5000, seed=42
):
    X_train, y_train, X_test, y_test = get_xy_traintest_specify(
        num_train,
        numbered_dataset_tag,
        layer,
        model_name,
        pos_ratio=0.5,
        MAX_AMT=MAX_AMT,
        seed=seed,
    )
    return X_train, y_train, X_test, y_test


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
    return points


def get_class_imbalance():
    min_size, max_size, num_points = 0.05, 0.95, 19
    points = np.linspace(min_size, max_size, num=num_points)
    return points


def get_classimabalance_num_train(numbered_dataset, min_num_test=100):
    # gives the maximum number of num_train possible
    y = get_yvals(numbered_dataset)
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
    return points


def get_OOD_datasets(translation=True):
    # translation tells us if we want that one living room dataset
    dataset_names = glob.glob("data/OOD data/*.csv")
    # Extract dataset names from paths by taking the base filename without _OOD.csv
    if translation:
        datasets = [
            os.path.basename(path).replace("_OOD.csv", "") for path in dataset_names
        ]
    else:
        datasets = [
            os.path.basename(path).replace("_OOD.csv", "")
            for path in dataset_names
            if "translation" not in path
        ]
    return datasets


def get_xy_OOD(dataset, model_name="gemma-2-9b", layer=20):
    X = torch.load(
        f"data/model_activations_{model_name}_OOD/{dataset}_OOD_blocks.{layer}.hook_resid_post.pt",
        weights_only=False,
    )
    df = pd.read_csv(f"data/OOD data/{dataset}_OOD.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["target"].values)
    return X, y


def get_OOD_traintest(dataset, model_name="gemma-2-9b", layer=20):
    X_train, y_train, _, _ = get_xy_traintest_specify(
        num_train=1024,
        numbered_dataset_tag=dataset,
        layer=layer,
        model_name=model_name,
        MAX_AMT=1500,
        pos_ratio=0.5,
        num_test=0,
    )
    X_test, y_test = get_xy_OOD(dataset, model_name, layer)
    return X_train, y_train, X_test, y_test


def get_xy_glue(toget="ensemble"):
    X = torch.load(
        "data/dataset_investigate/87_glue_cola_blocks.20.hook_resid_post.pt",
        weights_only=False,
    )
    df = pd.read_csv("results/investigate/87_glue_cola_investigate.csv")
    le = LabelEncoder()
    y = le.fit_transform(df[toget].values)
    return X, y


def get_disagree_glue(path_beginning=""):
    df = pd.read_csv(
        f"{path_beginning}/results/investigate/87_glue_cola_investigate.csv"
    )
    # Get indices where original target and ensemble predictions disagree
    original = np.array(df["original_target"], dtype=int)
    ensemble = np.array(df["ensemble"], dtype=int)
    disagree_idx = np.where(original != ensemble)[0]
    return disagree_idx


def get_glue_traintest(toget="ensemble", model_name="gemma-2-9b", layer=20):
    X_train, y_train, _, _ = get_xy_traintest_specify(
        num_train=1024,
        numbered_dataset_tag="87_glue_cola",
        layer=layer,
        model_name=model_name,
        MAX_AMT=1500,
        pos_ratio=0.5,
        num_test=0,
    )
    X_test, y_test = get_xy_glue(toget)
    return X_train, y_train, X_test, y_test


def get_datasets(model_name="llama-3.1-8b"):
    # Get all files in the directory
    dataset_sizes = get_dataset_sizes()
    files = os.listdir(f"data/model_activations_{model_name}")

    # Filter for files containing 'blocks'
    block_files = [f for f in files if "blocks" in f]

    # Extract unique dataset names by removing _blocks and everything after
    datasets = set()
    for file in block_files:
        dataset = file.split("_blocks")[0]
        if dataset in dataset_sizes.keys():  # binary dfs
            datasets.add(dataset)

    return sorted(list(datasets))


def get_layers(model_name="gemma-2-9b"):
    if model_name == "gemma-2-9b":
        layers = ["embed", 9, 20, 31, 41]
    elif model_name == "llama-3.1-8b":
        layers = ["embed", 8, 16, 24, 31]
    elif model_name == "gemma-2-2b":
        layers = [12]
    else:
        raise ValueError("model not accepted")
    return layers


def get_avg_test_size():
    sizes = get_dataset_sizes()
    test = []
    for dataset in sizes.keys():
        size = sizes[dataset]
        test.append(max(100, size - 1024))
    test = np.array(test)


# get_avg_test_size()
# print(len(get_datasets()))

# get_OOD_traintest('87_glue_cola')
# p = get_training_sizes()
# Xtrain, ytrain, Xtest, ytest = get_glue_traintest()
# print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)

# dataset = '100_news_fake'
# num_train, num_test = get_classimabalance_num_train(dataset)
# # print(num_train)
# # # print(num_train)
# print(num_train, num_test)
# X_train, y_train, X_test, y_test  = get_xy_traintest_specify(num_train = num_train,num_test = num_test, numbered_dataset_tag = dataset, layer = 20, model_name = 'gemma-2-9b', pos_ratio=0.05)
# print(y_test.shape)
# # print(X_test.shape[0], X_train.shape[0])
# X_train, y_train, X_test, y_test  = get_xy_traintest(1024, dataset, layer = 20, model_name = 'gemma-2-9b')
# # print(f"Train ratio: {sum(y_train)/len(y_train)}, Test ratio: {sum(y_test)/len(y_test)}")
# print(y_train, y_train.sum())

# y_train2 = corrupt_ytrain(y_train, frac = 0.25)
# print(np.abs(y_train - y_train2).sum())

# points = get_corrupt_frac()
# print(points)
