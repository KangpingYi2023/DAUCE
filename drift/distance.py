"""The DAUCE drift distance.

    Dist(T_p, T_c) = (1 - w) * D_marginal + w * D_dependency

``D_marginal`` is a type-aware average of per-column divergences: numeric columns
use a moment-matched Gaussian surrogate, categorical columns use Jensen-Shannon
divergence over category frequencies. ``D_dependency`` covers the drift that
per-column statistics provably cannot see -- a joint distribution can change a
lot while every marginal stays put -- by comparing the coupling of the top-k most
associated column pairs.

Both terms land in [0, 1], so the convex combination is on one scale and ``w`` is
interpretable.

Scope. This is a lightweight detection statistic, not a full joint-density model.
It is blind to (i) dependencies of order three and higher, (ii) column pairs
outside the top-k selection, and (iii) shape changes that move neither the first
two moments of a numeric column nor the coupling of a selected pair.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from drift import stats as dstats
from drift.schema import ColumnKind

LN2 = float(np.log(2.0))

DEFAULT_WEIGHT = 0.5
DEFAULT_N_BINS = 16
DEFAULT_IPF_ITERS = 30
DEFAULT_PAIR_AGGREGATE = "mean"
DEFAULT_PAIR_TOP_Q = 0.25
PAIR_AGGREGATES = ("mean", "topq", "max")


# --------------------------------------------------------------------------- #
# Per-column divergences
# --------------------------------------------------------------------------- #
def symmetric_gaussian_kl(
        mean_p: np.ndarray,
        var_p: np.ndarray,
        mean_c: np.ndarray,
        var_c: np.ndarray,
        eps: float = 1e-12,
) -> np.ndarray:
    """Symmetrised KL divergence between moment-matched Gaussians.

    For p = N(mu_p, var_p) and c = N(mu_c, var_c) the two one-sided divergences
    are

        KL(p||c) = log(sd_c/sd_p) + (var_p + d^2) / (2 var_c) - 1/2
        KL(c||p) = log(sd_p/sd_c) + (var_c + d^2) / (2 var_p) - 1/2

    with d = mu_p - mu_c. Averaging them cancels the logarithms and leaves

        J = 1/2 [ (var_p + d^2) / (2 var_c) + (var_c + d^2) / (2 var_p) - 1 ]

    which is symmetric, free of the log-ratio's numerical trouble when a variance
    is tiny, and zero exactly when the two moment pairs agree.

    The Gaussian is a moment-matched surrogate, not an assumption that the data
    is Gaussian: it is the maximum-entropy distribution for a given mean and
    variance, so J quantifies how much the first two moments moved.
    """
    var_p = np.maximum(np.asarray(var_p, dtype=np.float64), eps)
    var_c = np.maximum(np.asarray(var_c, dtype=np.float64), eps)
    d_sq = (np.asarray(mean_p, dtype=np.float64) - np.asarray(mean_c, dtype=np.float64)) ** 2
    return 0.5 * ((var_p + d_sq) / (2.0 * var_c) + (var_c + d_sq) / (2.0 * var_p) - 1.0)


def normalize_kl(kl: np.ndarray) -> np.ndarray:
    """Squash an unbounded divergence into [0, 1] via ``1 - exp(-kl)``.

    Monotone and zero-preserving, so ordering is untouched, but it puts the
    numeric term on the same scale as the [0, 1] categorical and dependency
    terms, which is what makes the weighted sum meaningful. Saturates at 1 for
    large divergences, where the exact magnitude no longer matters.
    """
    return 1.0 - np.exp(-np.maximum(np.asarray(kl, dtype=np.float64), 0.0))


def jensen_shannon(counts_p: np.ndarray, counts_c: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats between two count vectors.

    Bounded by ln 2 and finite even where one side has zero mass, which is why it
    is used instead of KL for discrete supports: a category that appears only
    after the update contributes information rather than an infinity.
    """
    p = np.asarray(counts_p, dtype=np.float64)
    c = np.asarray(counts_c, dtype=np.float64)
    p_sum, c_sum = p.sum(), c.sum()
    if p_sum <= 0 or c_sum <= 0:
        return 0.0
    p = p / p_sum
    c = c / c_sum
    m = 0.5 * (p + c)
    ok = m > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term_p = np.sum(np.where(p[ok] > 0, p[ok] * np.log(p[ok] / m[ok]), 0.0))
        term_c = np.sum(np.where(c[ok] > 0, c[ok] * np.log(c[ok] / m[ok]), 0.0))
    return float(max(0.5 * (term_p + term_c), 0.0))


def normalized_js(counts_p: np.ndarray, counts_c: np.ndarray) -> float:
    """Jensen-Shannon divergence rescaled to [0, 1]."""
    return min(jensen_shannon(counts_p, counts_c) / LN2, 1.0)


# --------------------------------------------------------------------------- #
# Dependency term
# --------------------------------------------------------------------------- #
def mutual_information(table: np.ndarray) -> float:
    """Mutual information in nats of a contingency table."""
    total = table.sum()
    if total <= 0:
        return 0.0
    joint = table / total
    row = joint.sum(axis=1, keepdims=True)
    col = joint.sum(axis=0, keepdims=True)
    independent = row * col
    ok = (joint > 0) & (independent > 0)
    if not np.any(ok):
        return 0.0
    return float(np.sum(joint[ok] * np.log(joint[ok] / independent[ok])))


def ipf_align(
        joint: np.ndarray,
        target_row: np.ndarray,
        target_col: np.ndarray,
        iters: int = DEFAULT_IPF_ITERS,
        eps: float = 1e-12,
) -> np.ndarray:
    """Rescale ``joint`` to the given marginals by iterative proportional fitting.

    Alternately scaling rows and columns converges to the distribution closest to
    ``joint`` in KL that has the target marginals, leaving the coupling intact.
    Applying this to both snapshots with the *same* target marginals strips the
    marginal component out, so the residual difference is dependency change only
    -- which keeps ``D_dependency`` complementary to ``D_marginal`` instead of
    re-measuring what the marginal term already saw.
    """
    out = np.asarray(joint, dtype=np.float64) + eps
    tr = np.asarray(target_row, dtype=np.float64).ravel()
    tc = np.asarray(target_col, dtype=np.float64).ravel()
    for _ in range(iters):
        row_sum = out.sum(axis=1)
        out *= np.where(row_sum > eps, tr / np.maximum(row_sum, eps), 1.0)[:, None]
        col_sum = out.sum(axis=0)
        out *= np.where(col_sum > eps, tc / np.maximum(col_sum, eps), 1.0)[None, :]
    total = out.sum()
    return out / total if total > 0 else out


def select_top_pairs(
        data: np.ndarray,
        edges: Sequence[np.ndarray],
        k: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Rank column pairs by mutual information on the reference snapshot.

    Runs offline, once per reference snapshot, and the result is cached: the
    online path only touches the k selected pairs, so detection stays O(k * n)
    instead of O(m^2 * n).
    """
    n_cols = data.shape[1]
    if k is None:
        k = n_cols
    k = max(0, min(int(k), n_cols * (n_cols - 1) // 2))
    if k == 0:
        return []
    binned = [dstats.apply_bins(data[:, j], edges[j]) for j in range(n_cols)]
    sizes = [len(edges[j]) + 1 for j in range(n_cols)]
    scored = []
    for a in range(n_cols):
        for b in range(a + 1, n_cols):
            flat = binned[a].astype(np.int64) * sizes[b] + binned[b].astype(np.int64)
            table = np.bincount(flat, minlength=sizes[a] * sizes[b]).reshape(sizes[a], sizes[b])
            scored.append((mutual_information(table.astype(np.float64)), a, b))
    # Sort on (-mi, a, b) so ties resolve deterministically across runs.
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(a, b) for _, a, b in scored[:k]]


def aggregate_pairs(
        values: np.ndarray,
        method: str = DEFAULT_PAIR_AGGREGATE,
        top_q: float = DEFAULT_PAIR_TOP_Q,
) -> float:
    """Pool the per-pair distances into ``D_dependency``.

    Dependency drift is usually concentrated: shuffling a few columns moves a
    handful of pairs a lot and leaves the rest at noise. A plain mean over all
    tracked pairs therefore dilutes the signal by roughly the ratio of tracked to
    affected pairs -- measured at 7.5x on Census with four shuffled columns.
    ``topq`` averages the largest ``top_q`` fraction instead, which keeps the
    signal while still averaging away some noise; ``max`` keeps all of it and
    none of the averaging.

    The threshold is calibrated with the same aggregator, so the no-drift floor
    moves with it and the comparison stays honest.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if method == "mean":
        return float(values.mean())
    if method == "max":
        return float(values.max())
    if method == "topq":
        if not 0.0 < top_q <= 1.0:
            raise ValueError(f"pair_top_q must be in (0, 1], got {top_q}")
        n_keep = max(1, int(np.ceil(top_q * values.size)))
        return float(np.sort(values)[-n_keep:].mean())
    raise ValueError(f"unknown pair aggregate '{method}'; "
                     f"available: {PAIR_AGGREGATES}")


# --------------------------------------------------------------------------- #
# Configuration and the distance itself
# --------------------------------------------------------------------------- #
@dataclass
class DistanceConfig:
    """Hyper-parameters of the drift distance.

    Attributes:
        weight: ``w``, the share given to the dependency term.
        n_bins: bins per axis of the pairwise contingency tables.
        top_k: number of column pairs to track; ``None`` means one per column.
        ipf_iters: IPF sweeps used to align marginals.
        align_marginals: when False the dependency term compares raw joints and
            therefore overlaps with the marginal term. Kept for the ablation.
        categorical_as_numeric: when True categorical columns go through the
            Gaussian surrogate, reproducing the encoding-dependent behaviour the
            type-aware treatment replaces. Kept for the ablation.
        pair_aggregate: how the per-pair distances are pooled; see
            :func:`aggregate_pairs`.
        pair_top_q: fraction of pairs kept when ``pair_aggregate='topq'``.
        categorical_bins: how categorical columns are binned for the dependency
            term. ``'category'`` (default) gives each category its own bin and
            merges the rare tail into one, making the term encoding-invariant;
            ``'quantile'`` is the encoding-dependent ablation.
    """

    weight: float = DEFAULT_WEIGHT
    n_bins: int = DEFAULT_N_BINS
    top_k: Optional[int] = None
    ipf_iters: int = DEFAULT_IPF_ITERS
    align_marginals: bool = True
    categorical_as_numeric: bool = False
    pair_aggregate: str = DEFAULT_PAIR_AGGREGATE
    pair_top_q: float = DEFAULT_PAIR_TOP_Q
    categorical_bins: str = "category"

    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")
        if self.categorical_bins not in ("quantile", "category"):
            raise ValueError(f"categorical_bins must be 'quantile' or 'category', "
                             f"got '{self.categorical_bins}'")
        if self.pair_aggregate not in PAIR_AGGREGATES:
            raise ValueError(f"pair_aggregate must be one of {PAIR_AGGREGATES}, "
                             f"got '{self.pair_aggregate}'")


@dataclass
class DistanceBreakdown:
    """The distance together with the parts it is made of."""

    distance: float
    d_marginal: float
    d_dependency: float
    per_column: np.ndarray
    per_pair: np.ndarray
    pairs: List[Tuple[int, int]]


class DriftDistance:
    """Reference-fitted drift distance.

    ``fit`` does the offline work (binning scheme, pair selection, reference
    summary); ``evaluate`` scores a current snapshot against it.
    """

    def __init__(self, config: Optional[DistanceConfig] = None):
        self.config = config or DistanceConfig()
        self.kinds: List[ColumnKind] = []
        self.numeric_cols: List[int] = []
        self.categorical_cols: List[int] = []
        self.edges: List[np.ndarray] = []
        self.code_maps: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        self.pairs: List[Tuple[int, int]] = []
        self.reference: Optional[dstats.TableSummary] = None

    def fit(self, reference: np.ndarray, kinds: Sequence[ColumnKind]) -> "DriftDistance":
        cfg = self.config
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError(f"reference must be 2-D, got shape {ref.shape}")
        if len(kinds) != ref.shape[1]:
            raise ValueError(f"got {len(kinds)} column kinds for {ref.shape[1]} columns")

        self.kinds = list(kinds)
        if cfg.categorical_as_numeric:
            self.numeric_cols = list(range(ref.shape[1]))
            self.categorical_cols = []
        else:
            self.numeric_cols = [j for j, k in enumerate(self.kinds) if k is ColumnKind.NUMERIC]
            self.categorical_cols = [j for j, k in enumerate(self.kinds) if k is ColumnKind.CATEGORICAL]

        self.edges = [dstats.quantile_edges(ref[:, j], cfg.n_bins) for j in range(ref.shape[1])]
        self.code_maps = {}
        if cfg.categorical_bins == "category":
            for j in self.categorical_cols:
                self.code_maps[j] = self._fit_code_map(ref[:, j], cfg.n_bins)
                self.edges[j] = np.arange(self.code_maps[j][2]) + 0.5
        self.pairs = ([] if cfg.weight == 0.0
                      else select_top_pairs(self._bin_view(ref), self.edges, cfg.top_k))
        self.reference = self._summarise(ref)
        return self

    @staticmethod
    def _fit_code_map(column: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Frequency-ranked bins for one categorical column.

        Categories get their own bin in descending reference frequency; when the
        domain exceeds ``n_bins``, only those strictly more frequent than the
        ``n_bins``-th one keep a bin and the tail shares the last one. Ties are
        never split, so the partition depends on the frequencies alone and not
        on which integer a category happens to carry. Unseen codes fall into the
        shared bin.
        """
        values, counts = np.unique(column, return_counts=True)
        if values.size <= n_bins:
            keep = np.ones(values.size, dtype=bool)
        else:
            cut = np.sort(counts)[::-1][n_bins - 1]
            keep = counts > cut
        ranked = np.argsort(-counts[keep], kind="stable")
        kept_values = values[keep][ranked]
        order = np.argsort(kept_values)
        return kept_values[order], order.astype(np.float64), int(keep.sum())

    def _bin_view(self, data: np.ndarray) -> np.ndarray:
        """The table the dependency grids are built on."""
        if not self.code_maps:
            return data
        out = np.array(data, dtype=np.float64, copy=True)
        for j, (codes, bins, n_kept) in self.code_maps.items():
            col = data[:, j]
            pos = np.clip(np.searchsorted(codes, col), 0, codes.size - 1)
            out[:, j] = np.where(codes[pos] == col, bins[pos], float(n_kept))
        return out

    def _summarise(self, data: np.ndarray) -> dstats.TableSummary:
        summary = dstats.TableSummary.build(
            data, self.numeric_cols, self.categorical_cols,
            [] if self.code_maps else self.pairs,
            edges=self.edges, n_bins=self.config.n_bins,
        )
        if self.code_maps:
            view = self._bin_view(data)
            summary.grids = {(a, b): dstats.build_pair_grid(view, a, b, self.edges)
                             for a, b in self.pairs}
        return summary

    def summarize(self, current: np.ndarray) -> dstats.TableSummary:
        """Build an incrementally mergeable summary using the fitted bins."""
        if self.reference is None:
            raise RuntimeError("call fit() before summarize()")
        return self._summarise(np.asarray(current, dtype=np.float64))

    def evaluate(self, current: np.ndarray) -> DistanceBreakdown:
        if self.reference is None:
            raise RuntimeError("call fit() before evaluate()")
        return self.evaluate_summary(self.summarize(current))

    def evaluate_summary(self, current: dstats.TableSummary) -> DistanceBreakdown:
        """Score a pre-built summary, for the incremental path."""
        if self.reference is None:
            raise RuntimeError("call fit() before evaluate_summary()")
        ref = self.reference
        cfg = self.config

        per_column = np.zeros(len(self.kinds), dtype=np.float64)
        if self.numeric_cols:
            kl = symmetric_gaussian_kl(
                ref.moments.mean, ref.moments.var,
                current.moments.mean, current.moments.var,
            )
            per_column[self.numeric_cols] = normalize_kl(kl)
        for j in self.categorical_cols:
            support = dstats.shared_support(ref.categories[j], current.categories[j])
            per_column[j] = normalized_js(
                ref.categories[j].aligned_frequencies(support),
                current.categories[j].aligned_frequencies(support),
            )
        d_marginal = float(per_column.mean()) if per_column.size else 0.0

        per_pair = np.zeros(len(self.pairs), dtype=np.float64)
        for i, key in enumerate(self.pairs):
            per_pair[i] = self._pair_distance(ref.grids[key].table, current.grids[key].table)
        d_dependency = aggregate_pairs(per_pair, cfg.pair_aggregate, cfg.pair_top_q)

        w = cfg.weight if self.pairs else 0.0
        distance = (1.0 - w) * d_marginal + w * d_dependency
        return DistanceBreakdown(
            distance=distance,
            d_marginal=d_marginal,
            d_dependency=d_dependency,
            per_column=per_column,
            per_pair=per_pair,
            pairs=list(self.pairs),
        )

    def _pair_distance(self, ref_table: np.ndarray, cur_table: np.ndarray) -> float:
        ref_total, cur_total = ref_table.sum(), cur_table.sum()
        if ref_total <= 0 or cur_total <= 0:
            return 0.0
        p = ref_table / ref_total
        q = cur_table / cur_total
        if self.config.align_marginals:
            # Both snapshots are pushed onto the reference marginals, so what is
            # left to compare is the coupling.
            target_row, target_col = p.sum(axis=1), p.sum(axis=0)
            p = ipf_align(p, target_row, target_col, self.config.ipf_iters)
            q = ipf_align(q, target_row, target_col, self.config.ipf_iters)
        return normalized_js(p.ravel(), q.ravel())


def drift_distance(
        reference: np.ndarray,
        current: np.ndarray,
        kinds: Sequence[ColumnKind],
        config: Optional[DistanceConfig] = None,
) -> DistanceBreakdown:
    """One-shot convenience wrapper around :class:`DriftDistance`."""
    return DriftDistance(config).fit(reference, kinds).evaluate(current)


def legacy_column_kl(reference: np.ndarray, current: np.ndarray) -> float:
    """The pre-revision distance: mean per-column Gaussian KL, all columns numeric.

    Kept verbatim (including reading every column as numeric) so the "before"
    column of the detection experiments is a faithful baseline rather than a
    reconstruction.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    mean_p, std_p = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    mean_c, std_c = np.nanmean(cur, axis=0), np.nanstd(cur, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = (np.log(std_c / std_p)
              + (std_p ** 2 + (mean_p - mean_c) ** 2) / (2 * std_c ** 2)
              - 0.5)
    return float(np.mean(kl))
