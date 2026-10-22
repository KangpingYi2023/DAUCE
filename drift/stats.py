"""Incrementally maintainable sufficient statistics for the drift distance.

Every statistic the distance needs is an additive summary of the rows, so a newly
inserted batch can be folded in with ``merge`` instead of rescanning the table.
That is what makes the detector cheap enough to run on every update: the online
cost is O(n_delta * (m + k)) rather than O(n * m).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NumericMoments:
    """Count, sum and sum of squares per numeric column."""

    count: int
    total: np.ndarray  # shape (n_cols,)
    total_sq: np.ndarray  # shape (n_cols,)

    @staticmethod
    def from_data(data: np.ndarray) -> "NumericMoments":
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        return NumericMoments(
            count=x.shape[0],
            total=np.nansum(x, axis=0),
            total_sq=np.nansum(x * x, axis=0),
        )

    def merge(self, other: "NumericMoments") -> "NumericMoments":
        return NumericMoments(
            count=self.count + other.count,
            total=self.total + other.total,
            total_sq=self.total_sq + other.total_sq,
        )

    @property
    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros_like(self.total)
        return self.total / self.count

    @property
    def var(self) -> np.ndarray:
        """Population variance, clipped at zero against round-off."""
        if self.count == 0:
            return np.zeros_like(self.total)
        return np.maximum(self.total_sq / self.count - self.mean ** 2, 0.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


@dataclass
class CategoryCounts:
    """Value frequencies of one categorical column."""

    counts: Dict[float, int] = field(default_factory=dict)

    @staticmethod
    def from_data(column: np.ndarray) -> "CategoryCounts":
        values, freqs = np.unique(np.asarray(column), return_counts=True)
        return CategoryCounts({float(v): int(c) for v, c in zip(values, freqs)})

    def merge(self, other: "CategoryCounts") -> "CategoryCounts":
        merged = dict(self.counts)
        for value, freq in other.counts.items():
            merged[value] = merged.get(value, 0) + freq
        return CategoryCounts(merged)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def aligned_frequencies(self, support: Sequence[float]) -> np.ndarray:
        """Frequencies over a shared support, so two snapshots stay comparable."""
        return np.array([self.counts.get(v, 0) for v in support], dtype=np.float64)


def shared_support(a: CategoryCounts, b: CategoryCounts) -> List[float]:
    """Union of the two category sets.

    A category that only one snapshot has still carries drift information, and
    Jensen-Shannon divergence tolerates the resulting zero probabilities.
    """
    return sorted(set(a.counts) | set(b.counts))


def quantile_edges(column: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges from the reference snapshot's quantiles.

    The edges are computed once on the reference snapshot and reused for both
    snapshots; binning the current snapshot with its own quantiles would hide
    exactly the shifts we want to see. Duplicate edges are dropped, so a skewed
    column simply gets fewer than ``n_bins`` bins.
    """
    col = np.asarray(column, dtype=np.float64)
    col = col[~np.isnan(col)]
    if col.size == 0:
        return np.empty(0, dtype=np.float64)
    qs = np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1))
    return np.unique(qs)[1:-1]


def apply_bins(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values onto bin indices in ``[0, len(edges)]``."""
    return np.searchsorted(edges, np.asarray(column, dtype=np.float64), side="right")


@dataclass
class PairGrid:
    """Binning scheme plus contingency table of one column pair."""

    col_a: int
    col_b: int
    edges_a: np.ndarray
    edges_b: np.ndarray
    table: np.ndarray  # shape (n_bins_a, n_bins_b)

    @property
    def n_bins_a(self) -> int:
        return len(self.edges_a) + 1

    @property
    def n_bins_b(self) -> int:
        return len(self.edges_b) + 1

    def merge(self, other: "PairGrid") -> "PairGrid":
        if (self.col_a, self.col_b) != (other.col_a, other.col_b):
            raise ValueError("cannot merge grids of different column pairs")
        if self.table.shape != other.table.shape:
            raise ValueError("cannot merge grids with different binning schemes")
        return PairGrid(self.col_a, self.col_b, self.edges_a, self.edges_b,
                        self.table + other.table)


def contingency(
        data: np.ndarray,
        col_a: int,
        col_b: int,
        edges_a: np.ndarray,
        edges_b: np.ndarray,
) -> np.ndarray:
    """Counts of ``(col_a, col_b)`` over a fixed binning scheme."""
    ia = apply_bins(data[:, col_a], edges_a)
    ib = apply_bins(data[:, col_b], edges_b)
    n_a, n_b = len(edges_a) + 1, len(edges_b) + 1
    # bincount on the flattened index is markedly faster than histogram2d.
    flat = ia.astype(np.int64) * n_b + ib.astype(np.int64)
    return np.bincount(flat, minlength=n_a * n_b).reshape(n_a, n_b).astype(np.float64)


def build_pair_grid(
        data: np.ndarray,
        col_a: int,
        col_b: int,
        edges: Sequence[np.ndarray],
) -> PairGrid:
    ea, eb = edges[col_a], edges[col_b]
    return PairGrid(col_a, col_b, ea, eb, contingency(data, col_a, col_b, ea, eb))


@dataclass
class TableSummary:
    """Everything the drift distance reads off one snapshot."""

    n_rows: int
    edges: List[np.ndarray]
    moments: NumericMoments
    categories: Dict[int, CategoryCounts]
    grids: Dict[Tuple[int, int], PairGrid]

    @staticmethod
    def build(
            data: np.ndarray,
            numeric_cols: Sequence[int],
            categorical_cols: Sequence[int],
            pairs: Sequence[Tuple[int, int]],
            edges: Optional[Sequence[np.ndarray]] = None,
            n_bins: int = 16,
    ) -> "TableSummary":
        """Summarise ``data``.

        Args:
            edges: reference binning scheme; recomputed from ``data`` when None.
                Always pass the reference edges when summarising the current
                snapshot.
        """
        if edges is None:
            edges = [quantile_edges(data[:, j], n_bins) for j in range(data.shape[1])]
        edges = list(edges)
        return TableSummary(
            n_rows=data.shape[0],
            edges=edges,
            moments=NumericMoments.from_data(data[:, list(numeric_cols)])
            if len(numeric_cols) else NumericMoments(data.shape[0], np.empty(0), np.empty(0)),
            categories={j: CategoryCounts.from_data(data[:, j]) for j in categorical_cols},
            grids={(a, b): build_pair_grid(data, a, b, edges) for a, b in pairs},
        )

    def merge(self, other: "TableSummary") -> "TableSummary":
        """Fold an update batch's summary into this one.

        ``other`` must have been built with this summary's ``edges``.
        """
        if len(self.edges) != len(other.edges):
            raise ValueError("cannot merge summaries with different column counts")
        return TableSummary(
            n_rows=self.n_rows + other.n_rows,
            edges=self.edges,
            moments=self.moments.merge(other.moments),
            categories={j: c.merge(other.categories[j]) for j, c in self.categories.items()},
            grids={key: g.merge(other.grids[key]) for key, g in self.grids.items()},
        )
