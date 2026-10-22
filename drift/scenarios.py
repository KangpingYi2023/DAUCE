"""Labelled drift scenarios.

Pure functions: given a reference table and a seed they return an update batch and
nothing else -- no files written, no communicator touched. The sampling semantics
match ``end2end/data_updater.py`` so the standalone detection experiments and the
end-to-end runs exercise the same notion of drift.

``dependency`` is new. It resamples rows and then independently shuffles a subset of
columns within the batch, which leaves every one of those columns' value multiset
exactly as it was while destroying the coupling between them. It is the concrete
counterexample to per-column detection: the marginals are unchanged by
construction, so any detector that only looks at them is blind to it.

``marginal`` is its mirror image, added after E17bound found that every other
positive type here also destroys the coupling: it warps a column's marginal with a
rank-preserving map, so the coupling survives exactly and only the marginals move.
It is the counterexample to dependency-only detection.

``deletion`` / ``deletion_uniform`` / ``modification`` cover the other update
types: the current snapshot is a *transformed* reference rather than the
reference plus an appended batch. ``deletion`` removes every row inside a
randomly placed quantile band of one randomly chosen column -- the snapshot
analogue of ``DELETE WHERE col BETWEEN lo AND hi``, which shifts the remaining
marginal by construction. ``deletion_uniform`` drops a uniformly random subset
of rows and is a negative: the survivors are a hypergeometric sample of the
reference, so the distribution is preserved. ``modification`` applies the column
drift value-skew mechanism *in place* to a random subset of rows -- the
analogue of ``UPDATE t SET col = skewed_value WHERE id IN (...)`` -- keeping the
table size constant.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zlib import crc32

import numpy as np

# Drift types that carry no distribution change and act as negatives.
NO_DRIFT_TYPES = ("random", "sample", "deletion_uniform")

# Types whose current snapshot is a transformed reference rather than
# reference + appended batch. They have no batch, so labelling cannot use the
# cardinality-additivity shortcut and detection must read the whole snapshot.
NON_APPEND_TYPES = ("deletion", "deletion_uniform", "modification")


def sample_batch(data: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform resample with replacement: an update that carries no drift."""
    idx = rng.choice(data.shape[0], size=batch_size, replace=True)
    return data[np.sort(idx)]


def row_drift_batch(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        alpha: float,
) -> np.ndarray:
    """Draw the batch from a fraction ``alpha`` of the rows.

    Smaller alpha concentrates the batch on fewer source rows, i.e. more severe
    drift. Mirrors ``TupleSkewSampler``.
    """
    n_rows = data.shape[0]
    n_candidates = max(1, int(np.ceil(alpha * n_rows)))
    candidates = rng.choice(n_rows, size=n_candidates, replace=False)
    idx = rng.choice(candidates, size=batch_size, replace=True)
    return data[np.sort(idx)]


def column_drift_batch(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        alpha: float,
        n_drift_cols: int = 2,
) -> np.ndarray:
    """Resample rows, then restrict some columns to a narrow value set.

    Mirrors ``ValueSkewSampler``: a large marginal shift confined to a few columns.
    """
    n_rows, n_cols = data.shape
    n_drift_cols = max(1, min(n_drift_cols, n_cols))
    drift_cols = rng.choice(n_cols, size=n_drift_cols, replace=False)
    batch = data[np.sort(rng.choice(n_rows, size=batch_size, replace=True))].copy()
    for col in drift_cols:
        n_candidates = max(1, int(round(alpha * n_rows)))
        candidates = rng.choice(data[:, col], size=n_candidates, replace=False)
        batch[:, col] = rng.choice(candidates, size=batch_size, replace=True)
    return batch


def dependency_drift_batch(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        n_drift_cols: int = 2,
) -> np.ndarray:
    """Resample rows, then shuffle some columns independently within the batch.

    Each shuffled column keeps its exact value multiset, so the batch's marginals
    are identical to those of a plain resample; only the joint distribution moves.
    ``n_drift_cols`` grades the severity, and at least two columns are needed for
    the coupling to change at all.
    """
    n_rows, n_cols = data.shape
    n_drift_cols = max(2, min(n_drift_cols, n_cols))
    drift_cols = rng.choice(n_cols, size=n_drift_cols, replace=False)
    batch = data[np.sort(rng.choice(n_rows, size=batch_size, replace=True))].copy()
    for col in drift_cols:
        batch[:, col] = batch[rng.permutation(batch_size), col]
    return batch


def marginal_drift_batch(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        alpha: float,
        n_drift_cols: int = 2,
) -> np.ndarray:
    """Resample rows, then warp some columns' marginals with a RANK-PRESERVING map.

    The exact mirror image of ``dependency_drift_batch``, and the case the drift
    distance's marginal term exists for. ``dependency`` keeps the marginals and
    destroys the coupling; this keeps the coupling and moves the marginals.

    Every other positive type in this module destroys the coupling as a side
    effect, which makes them unable to tell the two distance terms apart:
    ``row`` concentrates the batch on few source rows (so all columns and all
    couplings move together) and ``column`` overwrites a column with values
    resampled INDEPENDENTLY of the other columns (line 84), which breaks the
    coupling just as surely as an explicit shuffle. E17bound measured the
    consequence: the dependency term wins on all three, including the ones meant
    to be marginal-only, so the grid could not show the marginal term earning its
    place. This type closes that gap.

    Mechanism: each drifted column's values are replaced by the reference
    column's empirical quantile at a warped rank ``u ** (1 / alpha)``. The map is
    monotone non-decreasing in the original value, so every row's ORDER within
    the column survives and the rank coupling with all other columns is
    preserved; only where the mass sits changes. Values are drawn from the
    reference column's own sorted values, so no unseen category is introduced and
    the categorical JS path stays well defined.

    ``alpha`` keeps this module's convention -- larger is milder, ``alpha = 1.0``
    is the identity map and therefore a null -- because the exponent is 1/alpha.
    """
    n_rows, n_cols = data.shape
    n_drift_cols = max(1, min(n_drift_cols, n_cols))
    drift_cols = rng.choice(n_cols, size=n_drift_cols, replace=False)
    batch = data[np.sort(rng.choice(n_rows, size=batch_size, replace=True))].copy()
    gamma = 1.0 / max(float(alpha), 1e-6)
    denom = max(batch_size - 1, 1)
    for col in drift_cols:
        ref_sorted = np.sort(data[:, col])
        # argsort of argsort gives ordinal ranks that are monotone in the value;
        # ties are broken arbitrarily but never inverted, so the map stays monotone.
        ranks = np.argsort(np.argsort(batch[:, col], kind="stable"), kind="stable")
        warped = (ranks / denom) ** gamma
        idx = np.clip(np.rint(warped * (n_rows - 1)).astype(np.int64), 0, n_rows - 1)
        batch[:, col] = ref_sorted[idx]
    return batch


def deletion_band_snapshot(
        data: np.ndarray,
        rng: np.random.Generator,
        band_width: float,
) -> np.ndarray:
    """Delete every row inside a random quantile band of a random column.

    Rank-based rather than value-based so the deleted count is exactly
    ``int(band_width * n)`` even on discrete columns, where a value range of
    width w could otherwise take anything from 0 to half the table. The
    survivors lose a contiguous slice of that column's marginal, which is the
    distribution shift a predicate DELETE leaves behind.
    """
    n_rows, n_cols = data.shape
    col = int(rng.integers(n_cols))
    q_start = float(rng.uniform(0.0, 1.0 - band_width))
    order = np.argsort(data[:, col], kind="stable")
    start = int(q_start * n_rows)
    count = max(1, int(band_width * n_rows))
    keep = np.ones(n_rows, dtype=bool)
    keep[order[start:start + count]] = False
    return data[keep]


def deletion_uniform_snapshot(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """Drop a uniformly random subset of rows: a distribution-preserving DELETE.

    The survivors are a hypergeometric sample of the reference, so this is the
    negative control for deletion semantics -- the counterpart of ``random``
    among the append types.
    """
    n_rows = data.shape[0]
    batch_size = min(batch_size, n_rows - 1)
    drop = rng.choice(n_rows, size=batch_size, replace=False)
    keep = np.ones(n_rows, dtype=bool)
    keep[drop] = False
    return data[keep]


def modification_snapshot(
        data: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        alpha: float,
        n_drift_cols: int = 2,
) -> np.ndarray:
    """Rewrite some columns of a random row subset in place, values drawn skewed.

    The in-place analogue of ``column_drift_batch``: same alpha-concentrated
    candidate pool per drifted column, applied as UPDATEs to ``batch_size``
    existing rows instead of as an appended batch. The table size is unchanged,
    so a detector that only watches row counts sees nothing.
    """
    n_rows, n_cols = data.shape
    batch_size = min(batch_size, n_rows)
    n_drift_cols = max(1, min(n_drift_cols, n_cols))
    drift_cols = rng.choice(n_cols, size=n_drift_cols, replace=False)
    rows = rng.choice(n_rows, size=batch_size, replace=False)
    out = data.copy()
    for col in drift_cols:
        n_candidates = max(1, int(round(alpha * n_rows)))
        candidates = rng.choice(data[:, col], size=n_candidates, replace=False)
        out[rows, col] = rng.choice(candidates, size=batch_size, replace=True)
    return out


@dataclass(frozen=True)
class Scenario:
    """A reproducible (reference, update batch) pair.

    Everything is derived from the fields, so the same scenario rebuilds
    identically in the labelling pass and in the detection pass.
    """

    dataset: str
    drift_type: str
    seed: int
    batch_size: int
    alpha: Optional[float] = None
    n_drift_cols: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def scenario_id(self) -> str:
        alpha = "na" if self.alpha is None else f"{self.alpha:g}"
        cols = "na" if self.n_drift_cols is None else str(self.n_drift_cols)
        return (f"{self.dataset}|{self.drift_type}|a{alpha}|c{cols}"
                f"|b{self.batch_size}|s{self.seed}")

    @property
    def by_construction_label(self) -> int:
        """1 when the generator was asked to introduce drift."""
        return 0 if self.drift_type in NO_DRIFT_TYPES else 1

    def make_batch(self, reference: np.ndarray) -> np.ndarray:
        """Build this scenario's update batch from the reference table."""
        if self.drift_type in NON_APPEND_TYPES:
            raise ValueError(f"drift type '{self.drift_type}' transforms the "
                             f"reference in place and has no batch; call "
                             f"make_snapshots instead")
        rng = self._rng()
        batch_size = min(self.batch_size, reference.shape[0])
        if self.drift_type in NO_DRIFT_TYPES:
            return sample_batch(reference, batch_size, rng)
        if self.drift_type == "row":
            if self.alpha is None:
                raise ValueError("row drift needs alpha")
            return row_drift_batch(reference, batch_size, rng, self.alpha)
        if self.drift_type == "column":
            if self.alpha is None:
                raise ValueError("column drift needs alpha")
            return column_drift_batch(reference, batch_size, rng, self.alpha,
                                      self.n_drift_cols or 2)
        if self.drift_type == "dependency":
            return dependency_drift_batch(reference, batch_size, rng,
                                          self.n_drift_cols or 2)
        if self.drift_type == "marginal":
            if self.alpha is None:
                raise ValueError("marginal drift needs alpha")
            return marginal_drift_batch(reference, batch_size, rng, self.alpha,
                                        self.n_drift_cols or 2)
        raise ValueError(f"unknown drift type '{self.drift_type}'")

    def make_snapshots(self, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(T_p, T_c)``: the reference and the post-update snapshot.

        Append types build ``T_c`` as reference + batch; deletion and
        modification types transform the reference in place, so ``T_c`` may be
        smaller than (deletion) or the same size as (modification) ``T_p``.
        """
        rng = self._rng()
        batch_size = min(self.batch_size, reference.shape[0])
        if self.drift_type == "deletion":
            if self.alpha is None:
                raise ValueError("deletion needs alpha (the band width)")
            return reference, deletion_band_snapshot(reference, rng, self.alpha)
        if self.drift_type == "deletion_uniform":
            return reference, deletion_uniform_snapshot(reference, batch_size, rng)
        if self.drift_type == "modification":
            if self.alpha is None:
                raise ValueError("modification needs alpha")
            return reference, modification_snapshot(reference, batch_size, rng,
                                                    self.alpha,
                                                    self.n_drift_cols or 2)
        batch = self.make_batch(reference)
        return reference, np.vstack((reference, batch))

    def _rng(self) -> np.random.Generator:
        # crc32 rather than hash(): the built-in string hash is salted per
        # process, which would make scenarios irreproducible across runs.
        return np.random.default_rng(crc32(self.scenario_id.encode("utf-8")))


def expand_scenarios(
        dataset: str,
        drift_types: Sequence[str],
        seeds: Sequence[int],
        batch_size: int,
        alphas: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
        n_drift_cols: Sequence[int] = (2,),
        no_drift_seeds: Optional[Sequence[int]] = None,
        alphas_by_type: Optional[Dict[str, Sequence[float]]] = None,
) -> List[Scenario]:
    """Enumerate the scenario grid, skipping parameters a drift type ignores.

    ``no_drift_seeds`` overrides ``seeds`` for the negatives, which need far more
    of them than the drift types do. The negatives are the only trials the
    false-positive rate is measured on, and it is compared against tau = 0.05: with
    nine of them the smallest non-zero rate expressible is already 0.111, so a
    correctly calibrated detector cannot help but look like it breaches tau.

    ``alphas_by_type`` overrides ``alphas`` for individual drift types, which is
    needed when one config mixes types whose severity scales live at different
    orders of magnitude (e.g. deletion band widths of 0.05/0.01 next to
    modification concentration factors of 0.01/0.001).
    """
    scenarios: List[Scenario] = []
    for drift_type in drift_types:
        type_seeds = seeds
        type_alphas = alphas
        if alphas_by_type and drift_type in alphas_by_type:
            type_alphas = alphas_by_type[drift_type]
        if drift_type in NO_DRIFT_TYPES and no_drift_seeds is not None:
            type_seeds = no_drift_seeds
        for seed in type_seeds:
            if drift_type in NO_DRIFT_TYPES:
                scenarios.append(Scenario(dataset, drift_type, seed, batch_size))
            elif drift_type == "row":
                for alpha in type_alphas:
                    scenarios.append(Scenario(dataset, drift_type, seed, batch_size,
                                              alpha=alpha))
            elif drift_type in ("column", "modification", "marginal"):
                for alpha in type_alphas:
                    for n_cols in n_drift_cols:
                        scenarios.append(Scenario(dataset, drift_type, seed, batch_size,
                                                  alpha=alpha, n_drift_cols=n_cols))
            elif drift_type == "deletion":
                # alpha is the quantile-band width here, not a concentration.
                for alpha in type_alphas:
                    scenarios.append(Scenario(dataset, drift_type, seed, batch_size,
                                              alpha=alpha))
            elif drift_type == "dependency":
                for n_cols in n_drift_cols:
                    scenarios.append(Scenario(dataset, drift_type, seed, batch_size,
                                              n_drift_cols=n_cols))
            else:
                raise ValueError(f"unknown drift type '{drift_type}'")
    return scenarios
