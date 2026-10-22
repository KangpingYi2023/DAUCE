"""Model-free dataset loading.

``update_utils.dataset_util`` wraps tables in ``Naru.common.CsvTable``, which pulls
in torch. The detection experiments only need a float matrix and the column kinds,
so they load the raw arrays directly and stay importable without torch.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from drift.schema import ColumnKind, canonical_dataset, infer_kinds

# Repository root, i.e. the DAUCE directory two levels up from this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Datasets whose .npy file carries the "_int" suffix.
_INT_SUFFIX_DATASETS = ("census",)


def dataset_path(dataset: str) -> str:
    """Resolve the raw .npy path of a dataset.

    Handles the two naming conventions in ``data/``: variants live beside their
    base dataset (``data/census/census_gaussian_int.npy``) and census files carry
    an ``_int`` suffix.
    """
    base = dataset.split("_")[0]
    suffix = "_int.npy" if base in _INT_SUFFIX_DATASETS else ".npy"
    return os.path.join(REPO_ROOT, "data", base, f"{dataset}{suffix}")


def stats_table_path(table_name: str) -> str:
    return os.path.join(REPO_ROOT, "data", "stats", "numpy", f"{table_name}.npy")


def available(dataset: str) -> bool:
    return os.path.isfile(dataset_path(dataset))


def load_dataset(
        dataset: str,
        max_rows: Optional[int] = None,
        dtype=np.float64,
) -> Tuple[np.ndarray, List[ColumnKind]]:
    """Load a dataset as a float matrix plus its column kinds.

    Args:
        dataset: e.g. ``census``, ``bjaq_gaussian``, or a stats table name.
        max_rows: keep only the first ``max_rows`` rows, for scale sweeps.

    Raises:
        FileNotFoundError: with the expected path, so a missing dataset (Power)
            fails loudly instead of being silently skipped.
    """
    path = dataset_path(dataset)
    if not os.path.isfile(path):
        alt = stats_table_path(dataset)
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"dataset '{dataset}' not found at {path}. Place the raw .npy "
                f"there, or drop the dataset from the experiment config."
            )
    data = np.load(path, allow_pickle=True)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array in {path}, got shape {data.shape}")
    data = np.ascontiguousarray(data, dtype=dtype)
    if max_rows is not None and max_rows < data.shape[0]:
        data = data[:max_rows]
    return data, infer_kinds(data, dataset=canonical_dataset(dataset))


def split_reference(
        data: np.ndarray,
        reference_fraction: float = 1.0,
) -> np.ndarray:
    """Take the leading rows as the reference snapshot.

    The default keeps the whole table, matching the pre-trained models: they were
    fitted on the full original table, so that table is the reference the stale
    model corresponds to.
    """
    if not 0.0 < reference_fraction <= 1.0:
        raise ValueError(f"reference_fraction must be in (0, 1], got {reference_fraction}")
    if reference_fraction == 1.0:
        return data
    return data[: int(data.shape[0] * reference_fraction)]
