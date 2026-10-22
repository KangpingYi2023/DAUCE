"""Data-only threshold calibration.

The pre-revision threshold was derived from a q-error bootstrap, which dragged the
query workload, the true cardinalities and the specific CE model into a detector
that is otherwise independent of all three. Calibration here uses nothing but the
reference snapshot.

The construction mimics the online statistic exactly. The reference table is split
into blocks the size of an update batch; holding out block ``B_i`` gives the pair
``(T_p \\ B_i, T_p)``, i.e. a reference snapshot extended by a batch that is drawn
from the same distribution and therefore carries no drift. The resulting scores
form an empirical no-drift distribution, and the split-conformal quantile of that
distribution is the threshold.

Under exchangeability, taking the ``ceil((1 - tau) * (K + 1))``-th smallest of K
calibration scores bounds the false-positive rate by ``tau``. That needs
``K >= 1/tau - 1`` disjoint blocks, so a large update batch relative to the table
leaves too few blocks for the guarantee to hold exactly -- see
:class:`CalibrationResult.exact` and the warning it carries.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

DEFAULT_TAU = 0.05
DEFAULT_REPEATS = 10

# Scoring callable: (reference_part, full_reference) -> distance
ScoreFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    threshold: float
    """Threshold to use; the pooled quantile when ``repeats > 1``."""

    strict_threshold: float
    """Single-partition split-conformal quantile, for the formal statement."""

    scores: np.ndarray
    """All no-drift calibration scores, pooled over repeats."""

    tau: float
    n_blocks: int
    block_size: int
    repeats: int
    exact: bool
    """Whether ``n_blocks`` supports the conformal guarantee at this ``tau``."""

    notes: List[str] = field(default_factory=list)

    @property
    def min_blocks_required(self) -> int:
        return min_blocks_for_tau(self.tau)

    def summary(self) -> str:
        kind = "exact" if self.exact else "approximate"
        return (f"delta={self.threshold:.6e} (strict={self.strict_threshold:.6e}), "
                f"tau={self.tau}, K={self.n_blocks}x{self.repeats} blocks of "
                f"{self.block_size} rows, {kind} conformal guarantee")


def min_blocks_for_tau(tau: float) -> int:
    """Smallest K for which the conformal quantile index stays within 1..K."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    return int(math.ceil(1.0 / tau)) - 1


def conformal_index(n_scores: int, tau: float) -> int:
    """Zero-based index of the ``ceil((1 - tau)(n + 1))``-th smallest score."""
    rank = math.ceil((1.0 - tau) * (n_scores + 1))
    return int(min(max(rank, 1), n_scores)) - 1


def conformal_quantile(scores: np.ndarray, tau: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("no calibration scores")
    return float(ordered[conformal_index(ordered.size, tau)])


def calibrate(
        reference: np.ndarray,
        batch_size: int,
        score_fn: ScoreFn,
        tau: float = DEFAULT_TAU,
        n_blocks: Optional[int] = None,
        repeats: int = DEFAULT_REPEATS,
        random_seed: int = 0,
        strict: bool = False,
) -> CalibrationResult:
    """Calibrate the drift threshold from the reference snapshot alone.

    Args:
        reference: the reference table ``T_p``.
        batch_size: expected number of rows per update batch; the calibration
            blocks are this size so the no-drift scores are on the same scale as
            the online statistic.
        score_fn: ``(reference_part, full_reference) -> distance``, the same
            statistic used online.
        tau: tolerated false-positive rate.
        n_blocks: number of disjoint blocks per partition; defaults to as many as
            the table allows, capped at ``min_blocks_for_tau(tau)``.
        repeats: independent re-partitions whose scores are pooled to stabilise
            the threshold. The strict single-partition value is returned too.
        random_seed: seed for the partitioning.
        strict: raise instead of warning when the block count cannot support the
            conformal guarantee.

    Returns:
        A :class:`CalibrationResult`.
    """
    ref = np.asarray(reference)
    n_rows = ref.shape[0]
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size >= n_rows:
        raise ValueError(f"batch_size {batch_size} must be smaller than the "
                         f"reference size {n_rows}")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    required = min_blocks_for_tau(tau)
    max_blocks = n_rows // batch_size
    if max_blocks < 2:
        raise ValueError(
            f"batch_size {batch_size} leaves only {max_blocks} block(s) in a "
            f"reference of {n_rows} rows; calibration needs at least 2"
        )

    notes: List[str] = []
    if n_blocks is None:
        n_blocks = min(max_blocks, required)
    else:
        n_blocks = int(n_blocks)
        if n_blocks > max_blocks:
            notes.append(f"n_blocks reduced from {n_blocks} to {max_blocks} "
                         f"(reference holds {n_rows} rows)")
            n_blocks = max_blocks
    n_blocks = max(n_blocks, 2)

    exact = n_blocks >= required
    if not exact:
        message = (
            f"tau={tau} needs at least {required} disjoint calibration blocks but "
            f"a batch of {batch_size} rows in a reference of {n_rows} rows allows "
            f"only {n_blocks}. The threshold is still the empirical "
            f"{1 - tau:.3f} quantile of the no-drift scores, but the conformal "
            f"false-positive guarantee is approximate. Use a smaller batch_size "
            f"or a larger tau for an exact guarantee."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        notes.append(message)

    rng = np.random.default_rng(random_seed)
    pooled: List[float] = []
    strict_scores: List[float] = []
    for r in range(repeats):
        order = rng.permutation(n_rows)
        round_scores = []
        for i in range(n_blocks):
            held_out = order[i * batch_size:(i + 1) * batch_size]
            keep = np.ones(n_rows, dtype=bool)
            keep[held_out] = False
            # (T_p \ B_i, T_p): the held-out block plays the part of a no-drift
            # update batch appended to the retained rows.
            round_scores.append(float(score_fn(ref[keep], ref)))
        pooled.extend(round_scores)
        if r == 0:
            strict_scores = round_scores

    scores = np.asarray(pooled, dtype=np.float64)
    return CalibrationResult(
        threshold=conformal_quantile(scores, tau),
        strict_threshold=conformal_quantile(np.asarray(strict_scores), tau),
        scores=scores,
        tau=tau,
        n_blocks=n_blocks,
        block_size=batch_size,
        repeats=repeats,
        exact=exact,
        notes=notes,
    )


def suggested_batch_size(n_rows: int, tau: float = DEFAULT_TAU) -> int:
    """Largest batch size that still admits an exact conformal guarantee."""
    return max(1, n_rows // min_blocks_for_tau(tau))
