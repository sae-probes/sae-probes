import numpy as np
import pandas as pd
import pytest

from sae_probes.utils_data import (
    corrupt_ytrain,
    get_binary_df,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_train_test_indices,
    get_training_sizes,
    get_yvals,
    read_dataset_df,
    read_numbered_dataset_df,
)


def test_get_binary_df():
    df = get_binary_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # Check that some binary datasets are found
    assert "Dataset Tag" in df.columns
    assert "Data type" in df.columns
    assert "Dataset save name" in df.columns
    assert all(df["Data type"] == "Binary Classification")
    # Check a specific known binary dataset
    assert "hist_fig_ismale" in df["Dataset Tag"].values
    assert (
        df[df["Dataset Tag"] == "hist_fig_ismale"]["Dataset save name"].iloc[0]  # type: ignore
        == "cleaned_data/5_hist_fig_ismale.csv"
    )


def test_get_numbered_binary_tags():
    tags = get_numbered_binary_tags()
    assert isinstance(tags, list)
    assert len(tags) > 0
    # Check for a specific known binary dataset tag (filename without extension)
    assert "5_hist_fig_ismale" in tags
    assert "87_glue_cola" in tags  # another example
    # Ensure no paths or extensions are present
    for tag in tags:
        assert "/" not in tag
        assert ".csv" not in tag


def test_read_dataset_df():
    # Test with a known binary dataset tag from the master CSV
    df = read_dataset_df("hist_fig_ismale")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "target" in df.columns  # All datasets should have a target column

    # Test with another dataset to be sure
    df_cola = read_dataset_df("glue_cola")
    assert isinstance(df_cola, pd.DataFrame)
    assert len(df_cola) > 0
    assert "target" in df_cola.columns


def test_read_numbered_dataset_df():
    # Test with a "numbered" version of a known dataset tag
    df = read_numbered_dataset_df(
        "5_hist_fig_ismale"
    )  # The number prefix is part of the filename
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "target" in df.columns

    df_cola = read_numbered_dataset_df("87_glue_cola")
    assert isinstance(df_cola, pd.DataFrame)
    assert len(df_cola) > 0
    assert "target" in df_cola.columns


def test_get_dataset_sizes():
    sizes = get_dataset_sizes()
    assert isinstance(sizes, dict)
    assert len(sizes) > 0
    # Check size for a known dataset (actual size might vary, so check it exists)
    assert "5_hist_fig_ismale" in sizes
    assert sizes["5_hist_fig_ismale"] > 0
    assert "87_glue_cola" in sizes
    assert sizes["87_glue_cola"] > 0


def test_get_yvals_returns_numpy_array():
    # Using a known dataset that should exist and have a 'target' column
    y = get_yvals("5_hist_fig_ismale")
    assert isinstance(y, np.ndarray)
    assert len(y) > 0
    # Check if values are 0s and 1s after LabelEncoder
    assert set(np.unique(y)).issubset({0, 1})

    y_cola = get_yvals("87_glue_cola")
    assert isinstance(y_cola, np.ndarray)
    assert len(y_cola) > 0
    assert set(np.unique(y_cola)).issubset({0, 1})


def test_get_train_test_indices_correct_split():
    y = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1] * 10
    )  # 140 samples, 70 pos, 70 neg
    num_train = 80
    num_test = 40
    pos_ratio = 0.5
    train_indices, test_indices = get_train_test_indices(
        y, num_train, num_test, pos_ratio, seed=42
    )

    assert len(train_indices) == num_train
    assert len(test_indices) == num_test
    assert len(set(train_indices).intersection(set(test_indices))) == 0  # No overlap

    # Check positive ratio in train set
    train_labels = y[train_indices]
    expected_pos_train = int(np.ceil(pos_ratio * num_train))
    assert np.sum(train_labels) == expected_pos_train

    # Check positive ratio in test set
    test_labels = y[test_indices]
    expected_pos_test = int(np.ceil(pos_ratio * num_test))
    assert np.sum(test_labels) == expected_pos_test


def test_get_train_test_indices_varied_ratios():
    y = np.array([0] * 80 + [1] * 20)  # 100 samples, 20 positive
    np.random.shuffle(y)
    num_train = 50
    num_test = 20

    # Test with low positive ratio
    train_idx_low, test_idx_low = get_train_test_indices(
        y, num_train, num_test, pos_ratio=0.1, seed=1
    )
    assert np.sum(y[train_idx_low]) == int(np.ceil(0.1 * num_train))
    assert np.sum(y[test_idx_low]) == int(np.ceil(0.1 * num_test))

    # Test with high positive ratio (if possible, otherwise it will take all available positives)
    # Need enough positives for this. Let's use y_balanced
    y_balanced = np.array([0] * 40 + [1] * 60)  # 100 samples, 60 positive, 40 negative
    np.random.shuffle(y_balanced)
    train_idx_high, test_idx_high = get_train_test_indices(
        y_balanced, num_train, num_test, pos_ratio=0.8, seed=2
    )
    assert np.sum(y_balanced[train_idx_high]) == int(
        np.ceil(0.8 * num_train)
    )  # 0.8 * 50 = 40
    assert np.sum(y_balanced[test_idx_high]) == int(
        np.ceil(0.8 * num_test)
    )  # 0.8 * 20 = 16


def test_get_train_test_indices_raises_value_error_if_not_enough_samples():
    y_small_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 neg, 2 pos
    with pytest.raises(ValueError):
        # Requesting 3 positive samples for training when only 2 are available
        get_train_test_indices(y_small_pos, num_train=6, num_test=2, pos_ratio=0.5)

    y_small_neg = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  # 8 pos, 2 neg
    with pytest.raises(ValueError):
        # Requesting 3 negative samples for training (pos_ratio=0.5 implies 3 neg for num_train=6) when only 2 are available
        get_train_test_indices(y_small_neg, num_train=6, num_test=2, pos_ratio=0.5)


def test_get_classimabalance_num_train():
    # We need to test this with actual data, as it calls get_yvals
    # "5_hist_fig_ismale" is a known binary dataset
    num_train, num_test = get_classimabalance_num_train("5_hist_fig_ismale")
    assert isinstance(num_train, int)
    assert isinstance(num_test, int)
    assert num_train > 0
    assert num_test >= 100  # default min_num_test

    y_vals = get_yvals("5_hist_fig_ismale")
    total_samples = len(y_vals)
    assert num_train + num_test <= total_samples

    # Test with another dataset
    num_train_cola, num_test_cola = get_classimabalance_num_train("87_glue_cola")
    assert isinstance(num_train_cola, int)
    assert isinstance(num_test_cola, int)
    assert num_train_cola > 0
    assert num_test_cola >= 100
    y_vals_cola = get_yvals("87_glue_cola")
    assert num_train_cola + num_test_cola <= len(y_vals_cola)


def test_get_training_sizes():
    sizes = get_training_sizes()
    assert isinstance(sizes, list)
    assert len(sizes) > 0
    assert all(size >= 2**1 for size in sizes)  # min_size = 1
    assert all(size <= 2**10 for size in sizes)  # max_size = 10
    assert len(np.unique(sizes)) == len(sizes)  # all points should be unique


def test_get_class_imbalance():
    points = get_class_imbalance()
    assert isinstance(points, list)
    assert len(points) == 19
    assert np.min(points) == 0.05
    assert np.max(points) == 0.95


def test_corrupt_ytrain_no_corruption():
    ytrain = np.array([0, 1, 0, 1, 0, 1])
    corrupted_ytrain = corrupt_ytrain(ytrain, 0.0)
    assert np.array_equal(ytrain, corrupted_ytrain)


def test_corrupt_ytrain_some_corruption():
    ytrain = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)  # 100 elements
    frac_to_corrupt = 0.2
    corrupted_ytrain = corrupt_ytrain(ytrain, frac_to_corrupt)
    assert not np.array_equal(ytrain, corrupted_ytrain)
    num_corrupted = np.sum(ytrain != corrupted_ytrain)
    assert num_corrupted == int(len(ytrain) * frac_to_corrupt)  # 20 elements


def test_corrupt_ytrain_max_corruption():
    ytrain = np.array([0, 1] * 50)  # 100 elements
    frac_to_corrupt = 0.5
    corrupted_ytrain = corrupt_ytrain(ytrain, frac_to_corrupt)
    assert not np.array_equal(ytrain, corrupted_ytrain)
    num_corrupted = np.sum(ytrain != corrupted_ytrain)
    assert num_corrupted == int(len(ytrain) * frac_to_corrupt)  # 50 elements


def test_corrupt_ytrain_all_same_initial():
    ytrain = np.zeros(100)
    frac_to_corrupt = 0.3
    corrupted_ytrain = corrupt_ytrain(ytrain, frac_to_corrupt)
    assert np.sum(corrupted_ytrain) == int(100 * frac_to_corrupt)


def test_get_corrupt_frac():
    fracs = get_corrupt_frac()
    assert isinstance(fracs, list)
    assert len(fracs) == 11
    assert np.min(fracs) == 0.0
    assert np.max(fracs) == 0.5


def test_get_OOD_datasets_with_translation():
    datasets = get_OOD_datasets(translation=True)
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    # Check for specific OOD dataset names (without _OOD.csv)
    # These names come from the files in sae_probes/data/OOD data/
    assert "5_hist_fig_ismale" in datasets  # e.g. 5_hist_fig_ismale_OOD.csv
    assert "87_glue_cola" in datasets  # e.g. 87_glue_cola_OOD.csv
    # The function get_OOD_datasets processes "66_living-room_OOD.csv" to "66_living-room"
    # and "66_living-room_translations.csv" to "66_living-room_translations.csv" (it only removes "_OOD.csv" if present).
    assert "66_living-room" in datasets  # from 66_living-room_OOD.csv
    assert (
        "66_living-room_translations.csv" in datasets
    )  # from 66_living-room_translations.csv

    # Example of another OOD dataset
    assert "7_hist_fig_ispolitician" in datasets  # from 7_hist_fig_ispolitician_OOD.csv


def test_get_OOD_datasets_without_translation():
    datasets = get_OOD_datasets(translation=False)
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "5_hist_fig_ismale" in datasets
    assert "87_glue_cola" in datasets
    # When translation is False, files with "translation" in their name are excluded
    assert "66_living-room_translations.csv" not in datasets
    assert (
        "66_living-room" in datasets
    )  # This one does not have "translation" in its name
