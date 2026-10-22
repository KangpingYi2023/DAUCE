import os
import sys
import math
import time
import copy
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("./")
from update_utils import path_util
from update_utils.end2end_utils import communicator
from update_utils.end2end_utils.json_communicator import JsonCommunicator


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


class Sampler(ABC):
    def __init__(
            self, update_fraction: float = 0.2, update_size: int = None, random_seed=1234
    ):
        self.update_fraction = update_fraction
        self.update_size = update_size
        self.random_seed = random_seed

    def get_update_size(self, data: np.ndarray):
        """
        Returns the specified update_size if given; otherwise, calculates update_size based on update_fraction
        """
        if self.update_size and self.update_size < data.shape[0] * 0.2:
            return self.update_size
        data_size = data.shape[0]
        self.update_size = int(data_size * self.update_fraction)
        return self.update_size

    @abstractmethod
    def sample(self, data: np.ndarray):
        pass


class PermuteSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.empty(shape=(self.update_size, n_cols))

        np.random.seed(self.random_seed)
        for i in range(self.update_size):
            if i % 100 == 0:
                print("Permute - {}/{}".format(i, self.update_size))
            idxs = np.random.choice(range(n_rows), n_cols, replace=False)
            for j, idx in enumerate(idxs):
                samples[i, j] = data[idx, j]

        # print("Permute - END")
        return samples.astype(np.float32)


class PermuteOptimizedSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.zeros((self.update_size, n_cols))

        np.random.seed(self.random_seed)
        for col in range(n_cols):
            samples[:, col] = np.random.choice(
                data[:, col], self.update_size, replace=True
            )

        # print("Permute - END")
        return samples


class SingleSamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("SingleSample - START")

        np.random.seed(self.random_seed)
        idx = np.random.randint(data.shape[0])
        sample_idx = [idx] * self.update_size
        sample = data[sample_idx]

        # print("SingleSample - END")
        return sample


class SamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("Sample - START")

        np.random.seed(self.random_seed)
        sample_idx = np.random.choice(
            range(data.shape[0]), size=self.update_size, replace=True
        )
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]

        # print("Sample - END")
        return sample


class TupleSkewSampler(Sampler):
    def __init__(
            self, update_fraction: float = 0.2, update_size: int = None, random_seed=1234, skew_ratio = None
    ):
        super().__init__(update_fraction, update_size, random_seed)
        self.skew_ratio = skew_ratio

    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("TupleSkewSample - START")

        # skew_ratio=1e-3
        if self.skew_ratio is None:
            self.skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))
        # print(skew_ratio)
        n_rows, n_cols = data.shape

        if not self.random_seed == "auto":
            np.random.seed(int(self.random_seed))
        candidate_idx = np.random.choice(n_rows, size=math.ceil(self.skew_ratio * n_rows), replace=False)
        sample_idx = np.random.choice(candidate_idx, size=self.update_size, replace=True)
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]

        # print(sample[:, column_idx])

        # print("TupleSkewSample - END")
        return sample


class ValueSkewSampler(Sampler):
    def __init__(
            self, update_fraction: float = 0.2, update_size: int = None, random_seed=1234, skew_ratio = None
    ):
        super().__init__(update_fraction, update_size, random_seed)
        self.skew_ratio = skew_ratio

    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        # print("ValueSkewSample - START")

        if self.skew_ratio is None:
            self.skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))

        # print(skew_ratio)
        n_rows, n_cols = data.shape
        max_n_cols = n_cols // 3
        min_n_cols = 1

        if not self.random_seed == "auto":
            np.random.seed(int(self.random_seed))
        # update_n_cols=np.random.randint(min_n_cols, max_n_cols)
        update_n_cols = 2
        # update_n_cols = max_n_cols
        update_col_idx = np.random.choice(range(n_cols), size=update_n_cols, replace=False)
        # update_col_idx=[0,2]

        sample_idx = np.random.choice(n_rows, size=self.update_size, replace=True)
        sample_idx = np.sort(sample_idx)
        samples = data[sample_idx]
        for update_col in update_col_idx:
            candidate_values = np.random.choice(data[:, update_col], size=round(self.skew_ratio * n_rows), replace=False)
            samples[:, update_col] = np.random.choice(candidate_values, size=self.update_size, replace=True)

        # print(sample[:, column_idx])

        # print("ValueSkewSample - END")
        return samples


class DependencyDriftSampler(Sampler):
    """Dependency-only drift: batch marginals preserved, coupling destroyed.

    Resamples rows, then independently shuffles a subset of columns within the
    batch. Each shuffled column keeps its exact value multiset, so the batch has
    the same marginals as a plain resample and only the joint distribution moves.

    This is the case a per-column detector cannot see, so it is what separates the
    revised drift distance from the pre-revision one. Delegates to
    ``drift.scenarios`` so the end-to-end runs and the standalone detection
    experiments generate the same thing.
    """

    def __init__(
            self, update_fraction: float = 0.2, update_size: int = None,
            random_seed=1234, n_drift_cols: int = None
    ):
        super().__init__(update_fraction, update_size, random_seed)
        self.n_drift_cols = n_drift_cols

    def sample(self, data: np.ndarray):
        from drift.scenarios import dependency_drift_batch

        self.get_update_size(data)
        n_drift_cols = self.n_drift_cols
        if n_drift_cols is None:
            n_drift_cols = int(JsonCommunicator().get_with_default(
                'data_updater.n_drift_cols', 2))
        seed = 1234 if self.random_seed == "auto" else int(self.random_seed)
        rng = np.random.default_rng(seed)
        return dependency_drift_batch(data, self.update_size, rng, n_drift_cols)


def create_sampler(
        sampler_type: str,
        update_fraction: float = 0.2,
        update_size: int = None,
        random_seed=1234,
        skew_ratio=None,
) -> Sampler:
    if sampler_type == "sample":
        return SamplingSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "permute":
        return PermuteOptimizedSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "tupleskew":
        return TupleSkewSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
            skew_ratio=skew_ratio,
        )
    if sampler_type == "valueskew":
        return ValueSkewSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
            skew_ratio=skew_ratio,
        )
    if sampler_type == "depdrift":
        return DependencyDriftSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    raise ValueError(f"Unknown sampler type: {sampler_type}")


class DataUpdater:
    def __init__(self, data: np.ndarray, sampler: Sampler):
        self.sampler = sampler
        self.raw_data = data
        self.sampled_data = None
        self.updated_data = None

    def get_sampled_data(self):
        """Return sampled data."""
        return self.sampled_data

    def get_updated_data(self):
        """Return updated data."""
        return self.updated_data

    def update_data(self):
        """Use sampler to sample data and merge with the raw data."""
        self.sampled_data = self.sampler.sample(self.raw_data)
        self.updated_data = np.vstack((self.raw_data, self.sampled_data))

    def store_updated_data_to_file(self, output_path: Path):
        """Save the updated data to the specified output path."""
        np.save(output_path, self.updated_data)

    @staticmethod
    def update_dataset_from_file_to_file(
            data_update_method: str,
            raw_dataset_path: Path,
            updated_dataset_path: Path,
            update_fraction: float = None,
            update_size: int = None,
            random_seed=1234,
    ):
        """Read the current dataset from the original path, update it, and save the updated dataset to a new path."""
        # Load the current dataset from the original path
        raw_data = np.load(raw_dataset_path, allow_pickle=True)  # Original data

        # Update the data
        updater = DataUpdater(
            data=raw_data,
            sampler=create_sampler(
                sampler_type=data_update_method,
                update_fraction=update_fraction,
                update_size=update_size,
                random_seed=random_seed,
            ),
        )  # Create DataUpdater
        updater.update_data()  # Perform data update
        sampled_data = updater.get_sampled_data()  # New data added

        # Save the updated data to the new path
        updater.store_updated_data_to_file(output_path=updated_dataset_path)

        # Calculate and save landmarks
        original_data_end = len(raw_data)
        updated_data_end = original_data_end + len(sampled_data)
        landmarks = [original_data_end, updated_data_end]
        communicator.SplitIndicesCommunicator().set(landmarks)  # Save landmarks to file

        return raw_data, sampled_data
    

    @staticmethod
    def update_dataset_from_dir_to_dir(
            table_name: str,
            data_update_method: str,
            raw_dataset_dir: Path,
            update_fraction: float = None,
            update_size: int = None,
            random_seed=1234,
    ):
        """Read the current dataset from the original path, update it, and save the updated dataset to a new path."""
        # Load the current dataset from the original path
        raw_csv_path = os.path.join(raw_dataset_dir, f"{table_name}.csv")
        raw_npy_path = os.path.join(raw_dataset_dir, f"{table_name}.npy") 
        raw_csv_data = pd.read_csv(raw_csv_path)  # Original data
        raw_npy_data = np.load(raw_npy_path, allow_pickle=True)

        # Update the data
        updater = DataUpdater(
            data=raw_csv_data.to_numpy(),
            sampler=create_sampler(
                sampler_type=data_update_method,
                update_fraction=update_fraction,
                update_size=update_size,
                random_seed=random_seed,
            ),
        )  # Create DataUpdater
        updater.update_data()  # Perform data update
        sampled_data = updater.get_sampled_data()  # New data added
        delta_csv = pd.DataFrame(sampled_data, columns=raw_csv_data.columns)
        updated_csv = pd.concat([raw_csv_data, delta_csv], ignore_index=True)
        updated_csv.to_csv(raw_csv_path, index=False)
        
        convert_sampled_data = copy.deepcopy(sampled_data)
        for idx, attr in enumerate(raw_csv_data.columns):
            if "Date" in attr:
                for i, date in enumerate(convert_sampled_data[:, idx]):
                    convert_date = timestamp_transorform(date)
                    convert_sampled_data[i][idx]=convert_date
        
        # convert_sampled_data[np.isnan(convert_sampled_data)] = 0
        convert_sampled_data = convert_sampled_data.astype(np.float64)

        # Save the updated data to the new path
        updated_npy_data = np.vstack((raw_npy_data, convert_sampled_data))
        np.save(raw_npy_path, updated_npy_data)

        # Calculate and save landmarks
        original_data_end = len(raw_npy_data)
        updated_data_end = original_data_end + len(sampled_data)
        landmarks = [original_data_end, updated_data_end]
        communicator.SplitIndicesCommunicator().set(landmarks)  # Save landmarks to file

        return raw_npy_data, convert_sampled_data, delta_csv


if __name__ == "__main__":
    # config
    dataset_name = "bjaq"
    assert dataset_name in ["census", "forest", "bjaq", "power"]
    raw_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
    # raw_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    new_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    abs_raw_file_path = path_util.get_absolute_path(raw_file_path)
    abs_new_file_path = path_util.get_absolute_path(new_file_path)

    # load raw npy data
    raw_data = np.load(abs_raw_file_path)
    print(raw_data.shape)

    # sampler
    samplers = [
        # PermuteOptimizedSampler(update_fraction=0.2),
        # SingleSamplingSampler(update_fraction=0.2),
        # SamplingSampler(update_fraction=0.2),
        # ValueSamplingSampler(update_fraction=0.2),
        # TupleSkewSampler(update_fraction=0.2)
        ValueSkewSampler(update_fraction=0.2)
    ]

    # updater
    for sampler in samplers:
        print("New sampler")
        updater = DataUpdater(raw_data, sampler=sampler)
        updater.update_data()
        updated_data = updater.get_updated_data()
        print(updated_data[:, 4])
        updater.store_updated_data_to_file(abs_new_file_path)
