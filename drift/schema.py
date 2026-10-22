"""Column-type metadata for drift detection.

The drift distance treats numeric and categorical columns differently: numeric
columns are summarised by their first two moments, categorical columns by their
category frequencies. Reading an integer-encoded categorical column as a numeric
one lets the arbitrary encoding order dictate the mean and variance, so the
column kinds have to be known before any statistic is computed.

Kinds come from an explicit per-dataset declaration (the DBMS catalog knows them
in a real deployment). ``infer_kinds`` only guesses for datasets that have no
declaration.
"""

from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np


class ColumnKind(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


# Column names in the physical order used by the .npy / .csv files, together with
# the names that are genuinely categorical. Indices are derived from the names so
# that a column-order change cannot silently shift the kinds.
DATASET_COLUMNS: Dict[str, List[str]] = {
    "census": [
        "age", "workclass", "fnlwgt", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "native-country",
    ],
    "bjaq": ["PM2.5", "PM10", "NO2", "O3", "TEMP"],
    "forest": [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ],
    # Power: data/power/power.npy (2049280 x 6, float64) is in the repository and
    # matches these six continuous columns; E1 detection covers it via
    # experiments/configs/E1_power.yaml.
    "power": ["c0", "c1", "c2", "c3", "c4", "c5"],
    "stats_badges": ["Id", "UserId", "Date"],
    "stats_votes": ["Id", "PostId", "VoteTypeId", "CreationDate", "UserId",
                    "BountyAmount"],
    "stats_postHistory": ["Id", "PostHistoryTypeId", "PostId", "CreationDate",
                          "UserId"],
    "stats_posts": ["Id", "PostTypeId", "CreationDate", "Score", "ViewCount",
                    "OwnerUserId", "AnswerCount", "CommentCount",
                    "FavoriteCount", "LastEditorUserId"],
    "stats_users": ["Id", "Reputation", "CreationDate", "Views", "UpVotes",
                    "DownVotes"],
    "stats_comments": ["Id", "PostId", "Score", "CreationDate", "UserId"],
    "stats_postLinks": ["Id", "CreationDate", "PostId", "RelatedPostId",
                        "LinkTypeId"],
    "stats_tags": ["Id", "Count", "ExcerptPostId"],
}

CATEGORICAL_COLUMNS: Dict[str, List[str]] = {
    "census": ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"],
    "bjaq": [],
    "forest": [],
    "power": [],
    "stats_badges": [],
    "stats_votes": ["VoteTypeId"],
    "stats_postHistory": ["PostHistoryTypeId"],
    "stats_posts": ["PostTypeId"],
    "stats_users": [],
    "stats_comments": [],
    "stats_postLinks": ["LinkTypeId"],
    "stats_tags": [],
}

# A column with few distinct integer values is treated as categorical when the
# dataset has no declaration. Deliberately conservative: Forest's Slope has 67
# distinct values and must stay numeric.
INFER_MAX_DISTINCT = 64


def canonical_dataset(dataset: str) -> str:
    """Map an experiment dataset name onto its schema key.

    ``census_gaussian`` and ``census`` share a schema, and per-table stats names
    arrive both as ``stats_posts`` and as a bare ``posts``.
    """
    if dataset in DATASET_COLUMNS:
        return dataset
    if dataset.endswith("_gaussian"):
        base = dataset[: -len("_gaussian")]
        if base in DATASET_COLUMNS:
            return base
    stats_key = f"stats_{dataset}"
    if stats_key in DATASET_COLUMNS:
        return stats_key
    return dataset


def column_names(dataset: str) -> Optional[List[str]]:
    return DATASET_COLUMNS.get(canonical_dataset(dataset))


def declared_kinds(dataset: str, n_cols: Optional[int] = None) -> Optional[List[ColumnKind]]:
    """Return the declared kinds, or None when the dataset is not declared."""
    key = canonical_dataset(dataset)
    names = DATASET_COLUMNS.get(key)
    if names is None:
        return None
    if n_cols is not None and n_cols != len(names):
        raise ValueError(
            f"dataset '{dataset}' is declared with {len(names)} columns "
            f"but the data has {n_cols}; update drift/schema.py"
        )
    categorical = set(CATEGORICAL_COLUMNS.get(key, []))
    unknown = categorical - set(names)
    if unknown:
        raise ValueError(f"dataset '{dataset}' marks unknown columns as categorical: {sorted(unknown)}")
    return [ColumnKind.CATEGORICAL if name in categorical else ColumnKind.NUMERIC
            for name in names]


def infer_kinds(
        data: np.ndarray,
        dataset: Optional[str] = None,
        max_distinct: int = INFER_MAX_DISTINCT,
) -> List[ColumnKind]:
    """Resolve the kind of every column, preferring the declaration.

    Args:
        data: 2-D array of shape (n_rows, n_cols).
        dataset: dataset name; when it is declared in this module the
            declaration wins and no inference happens.
        max_distinct: inference cut-off for integer-valued columns.
    """
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {data.shape}")
    n_cols = data.shape[1]

    if dataset is not None:
        kinds = declared_kinds(dataset, n_cols=n_cols)
        if kinds is not None:
            return kinds

    kinds = []
    for j in range(n_cols):
        col = data[:, j]
        col = col[~np.isnan(col)] if np.issubdtype(col.dtype, np.floating) else col
        integral = col.size == 0 or np.all(col == np.floor(col))
        n_distinct = np.unique(col).size
        is_cat = bool(integral and n_distinct <= max_distinct)
        kinds.append(ColumnKind.CATEGORICAL if is_cat else ColumnKind.NUMERIC)
    return kinds


def numeric_indices(kinds: Sequence[ColumnKind]) -> List[int]:
    return [j for j, k in enumerate(kinds) if k is ColumnKind.NUMERIC]


def categorical_indices(kinds: Sequence[ColumnKind]) -> List[int]:
    return [j for j, k in enumerate(kinds) if k is ColumnKind.CATEGORICAL]


def describe(dataset: str, kinds: Sequence[ColumnKind]) -> str:
    names = column_names(dataset)
    if names is None:
        names = [f"col{j}" for j in range(len(kinds))]
    parts = [f"{n}:{k.value[:3]}" for n, k in zip(names, kinds)]
    return f"{dataset} [{', '.join(parts)}]"
