"""Drift detectors behind one interface.

Every detector reduces a pair of snapshots to a scalar distance and gets its
threshold from :mod:`drift.calibration`, so comparisons between them are about the
statistic and never about a hand-tuned cut-off.

Nothing here imports torch or loads a CE model: the detectors are strictly
model-agnostic. The loss-based DDUp detector is the exception and stays in
``Naru/eval_model.py`` where the models live.

Baselines that summarise a subsample (GMM, MMD, sliced Wasserstein) take a
``sample_budget``. Their accuracy and their cost both move with that budget, so
report them as a curve over budgets rather than a single point.

Subsampling is re-randomised on every draw and each such statistic is averaged
over ``subsample_repeats`` draws. Both are required for :mod:`drift.calibration`
to mean anything on these baselines: a frozen subsample makes the offline and the
online score two unrelated single realisations of a noisy quantity rather than
exchangeable draws, and on Bjaq that noise spans 20x -- an order of magnitude more
than the drift it is supposed to measure.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from drift import calibration
from drift.distance import (DEFAULT_IPF_ITERS, DEFAULT_N_BINS,
                            DEFAULT_PAIR_AGGREGATE, DEFAULT_PAIR_TOP_Q,
                            DEFAULT_WEIGHT, DistanceConfig, DriftDistance,
                            legacy_column_kl)
from drift.schema import ColumnKind

DEFAULT_SUBSAMPLE_REPEATS = 5
"""Independent subsample draws averaged into one score.

Five is where the Bjaq no-drift spread stops dominating the drift signal; it costs
the subsampling baselines a factor of five in latency, reported as such in E7.
"""


@dataclass
class DetectionResult:
    """One detection decision."""

    distance: float
    threshold: Optional[float]
    is_drift: Optional[bool]
    latency_s: float
    extras: Dict[str, Any] = field(default_factory=dict)


class Detector:
    """Base class: fit on the reference snapshot, then score current snapshots."""

    name = "detector"

    def __init__(self, **params):
        self.params = params
        self._calibration: Optional[calibration.CalibrationResult] = None
        self._scale_mean: Optional[np.ndarray] = None
        self._scale_std: Optional[np.ndarray] = None
        self._draws = 0

    # -- offline ----------------------------------------------------------- #
    def fit_reference(self, reference: np.ndarray, kinds: Sequence[ColumnKind]) -> "Detector":
        raise NotImplementedError

    # -- online ------------------------------------------------------------ #
    def distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        raise NotImplementedError

    # -- threshold --------------------------------------------------------- #
    def calibrate(
            self,
            reference: np.ndarray,
            kinds: Sequence[ColumnKind],
            batch_size: int,
            tau: float = calibration.DEFAULT_TAU,
            n_blocks: Optional[int] = None,
            repeats: int = calibration.DEFAULT_REPEATS,
            random_seed: int = 0,
    ) -> calibration.CalibrationResult:
        """Derive the threshold from no-drift blocks of the reference snapshot.

        Each block is scored by a detector refitted on the retained rows, matching
        what happens online where the reference is whatever is already in the
        table.
        """
        def score_fn(ref_part: np.ndarray, full_ref: np.ndarray) -> float:
            # Hand the draw counter to the probe and take it back, so the blocks
            # continue one seed stream instead of each restarting it. A fresh
            # probe starts at draw zero, and every block hands ``_subsample`` the
            # same two array sizes, so without this the blocks would all pick the
            # same rows: nineteen replicas of one frozen subsample, with a spread
            # far below the online one and no exchangeability with it.
            probe = self.__class__(**self.params)
            probe._draws = self._draws
            probe.fit_reference(ref_part, kinds)
            dist = probe.distance(ref_part, full_ref)
            self._draws = probe._draws
            return dist

        self._calibration = calibration.calibrate(
            reference=reference,
            batch_size=batch_size,
            score_fn=score_fn,
            tau=tau,
            n_blocks=n_blocks,
            repeats=repeats,
            random_seed=random_seed,
        )
        return self._calibration

    @property
    def threshold(self) -> Optional[float]:
        return self._calibration.threshold if self._calibration else None

    def detect(self, reference: np.ndarray, current: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        dist = self.distance(reference, current)
        latency = time.perf_counter() - start
        threshold = self.threshold
        return DetectionResult(
            distance=dist,
            threshold=threshold,
            is_drift=None if threshold is None else bool(dist > threshold),
            latency_s=latency,
        )

    def _next_seed(self) -> int:
        """A fresh seed for every draw, derived from ``random_seed``.

        Re-randomising matters more than it looks. Online the two snapshots always
        have the same pair of row counts, so a fixed seed makes ``_subsample`` pick
        the same positions in every scenario: what reads as N independent
        no-drift trials is one frozen draw repeated N times. Calibration draws its
        own frozen realisation from differently sized arrays, and the two are then
        not exchangeable, which is exactly what voids the conformal bound. Seeding
        off ``random_seed`` plus a draw counter keeps a whole run reproducible
        while making successive draws independent.
        """
        self._draws += 1
        base = int(getattr(self, "random_seed", 0))
        return int(np.random.default_rng([base, self._draws]).integers(1 << 31))

    def _subsample(self, data: np.ndarray, budget: Optional[int]) -> np.ndarray:
        """Draw at most ``budget`` rows without replacement, freshly each call."""
        data = np.asarray(data)
        if budget is None or budget >= data.shape[0]:
            return data
        rng = np.random.default_rng(self._next_seed())
        idx = rng.choice(data.shape[0], size=int(budget), replace=False)
        return data[idx]

    def _effective_repeats(self, repeats: int, budget: Optional[int],
                           *arrays: np.ndarray) -> int:
        """Collapse to a single draw when the budget subsamples nothing.

        Averaging identical draws would only multiply the cost, so a budget that
        already covers both snapshots keeps the detector at its original price.
        """
        if budget is None or all(budget >= np.asarray(a).shape[0] for a in arrays):
            return 1
        return max(1, int(repeats))

    def _averaged(self, draw, repeats: int) -> float:
        """Mean of ``repeats`` independent draws of a subsampled statistic.

        Averaging shrinks the subsampling variance without changing what the
        statistic estimates, and it is applied identically offline and online so
        the calibration scores stay exchangeable with the online one.
        """
        if repeats <= 1:
            return float(draw())
        return float(np.mean([draw() for _ in range(repeats)]))

    def _fit_scaler(self, reference: np.ndarray) -> None:
        """Remember the reference scale so both snapshots can be standardised.

        Columns of raw tables differ by orders of magnitude (Census fnlwgt spans
        ~1e6 while sex is 0/1). Left unscaled, a covariance regulariser or a
        kernel bandwidth is meaningless on most columns. Standardising with the
        *reference* statistics is a fixed affine map, so a shift in the current
        snapshot survives it -- unlike pooled scaling, which would partly absorb
        the very drift being measured.
        """
        ref = np.asarray(reference, dtype=np.float64)
        self._scale_mean = ref.mean(axis=0)
        scale = ref.std(axis=0)
        self._scale_std = np.where(scale > 0, scale, 1.0)

    def _standardize(self, data: np.ndarray) -> np.ndarray:
        if getattr(self, "_scale_mean", None) is None:
            return np.asarray(data, dtype=np.float64)
        return (np.asarray(data, dtype=np.float64) - self._scale_mean) / self._scale_std


# --------------------------------------------------------------------------- #
# DAUCE
# --------------------------------------------------------------------------- #
class DauceDetector(Detector):
    """The revised DAUCE distance: type-aware marginals plus top-k dependency."""

    name = "dauce"

    def __init__(
            self,
            weight: float = DEFAULT_WEIGHT,
            n_bins: int = DEFAULT_N_BINS,
            top_k: Optional[int] = None,
            ipf_iters: int = DEFAULT_IPF_ITERS,
            align_marginals: bool = True,
            categorical_as_numeric: bool = False,
            pair_aggregate: str = DEFAULT_PAIR_AGGREGATE,
            pair_top_q: float = DEFAULT_PAIR_TOP_Q,
            sample_budget: Optional[int] = None,
            categorical_bins: str = "category",
    ):
        super().__init__(weight=weight, n_bins=n_bins, top_k=top_k,
                         ipf_iters=ipf_iters, align_marginals=align_marginals,
                         categorical_as_numeric=categorical_as_numeric,
                         pair_aggregate=pair_aggregate, pair_top_q=pair_top_q,
                         sample_budget=sample_budget,
                         **({} if categorical_bins == "category"
                            else {"categorical_bins": categorical_bins}))
        self.config = DistanceConfig(
            weight=weight, n_bins=n_bins, top_k=top_k, ipf_iters=ipf_iters,
            align_marginals=align_marginals,
            categorical_as_numeric=categorical_as_numeric,
            pair_aggregate=pair_aggregate, pair_top_q=pair_top_q,
            categorical_bins=categorical_bins,
        )
        self.sample_budget = sample_budget
        self._distance = DriftDistance(self.config)
        self.last_breakdown = None

    def fit_reference(self, reference, kinds):
        self._distance.fit(reference, kinds)
        return self

    def distance(self, reference, current):
        # Budget ablation: cap the online snapshot at ``sample_budget`` rows.
        # The draw is fresh and seeded off the draw counter, so calibration
        # blocks and online scores stay exchangeable. The reference fit stays
        # on the full table: it is a one-time offline cost and the reference
        # is the table the pre-trained model corresponds to.
        current = self._subsample(np.asarray(current), self.sample_budget)
        breakdown = self._distance.evaluate(current)
        self.last_breakdown = breakdown
        return breakdown.distance

    def detect(self, reference, current):
        result = super().detect(reference, current)
        if self.last_breakdown is not None:
            result.extras["d_marginal"] = self.last_breakdown.d_marginal
            result.extras["d_dependency"] = self.last_breakdown.d_dependency
            result.extras["n_pairs"] = len(self.last_breakdown.pairs)
        return result


class LegacyGaussianKLDetector(Detector):
    """The pre-revision statistic: mean per-column Gaussian KL, all columns numeric."""

    name = "legacy_kl"

    def fit_reference(self, reference, kinds):
        return self

    def distance(self, reference, current):
        return legacy_column_kl(reference, current)


# --------------------------------------------------------------------------- #
# Structural baselines
# --------------------------------------------------------------------------- #
class GMMJointDetector(Detector):
    """Joint Gaussian mixture on all columns, compared by Monte-Carlo JS.

    The universal-approximator answer to "why not model the joint density
    properly": fit a full-covariance mixture to each snapshot and estimate the
    Jensen-Shannon divergence between the two fits by sampling. Faithful but
    costly, and the EM plus Monte-Carlo noise floor is high enough to blunt its
    sensitivity at a fixed budget.
    """

    name = "gmm_joint"

    def __init__(self, n_components: int = 5, sample_budget: Optional[int] = 50000,
                 mc_samples: int = 20000, reg_covar: float = 1e-4,
                 max_iter: int = 100, random_seed: int = 0,
                 subsample_repeats: int = DEFAULT_SUBSAMPLE_REPEATS):
        super().__init__(n_components=n_components, sample_budget=sample_budget,
                         mc_samples=mc_samples, reg_covar=reg_covar,
                         max_iter=max_iter, random_seed=random_seed,
                         subsample_repeats=subsample_repeats)
        self.n_components = n_components
        self.sample_budget = sample_budget
        self.mc_samples = mc_samples
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.subsample_repeats = subsample_repeats
        self._reference_model = None

    def _fit_gmm(self, data: np.ndarray, warm_start=None):
        from sklearn.mixture import GaussianMixture
        seed = self._next_seed()
        sample = self._subsample(np.asarray(data, dtype=np.float64), self.sample_budget)
        n_components = max(1, min(self.n_components, sample.shape[0]))
        init = {}
        if warm_start is not None and warm_start.n_components == n_components:
            init = dict(means_init=warm_start.means_,
                        weights_init=warm_start.weights_,
                        precisions_init=warm_start.precisions_)
        model = GaussianMixture(
            n_components=n_components, covariance_type="full",
            reg_covar=self.reg_covar, max_iter=self.max_iter, random_state=seed,
            **init,
        )
        return model.fit(sample)

    def fit_reference(self, reference, kinds):
        # Standardise first: reg_covar is an absolute floor, so on raw columns
        # spanning several orders of magnitude the fit degenerates and the
        # Monte-Carlo JS estimate saturates at its ceiling even without drift.
        self._fit_scaler(reference)
        self._reference_model = self._fit_gmm(self._standardize(reference))
        return self

    def distance(self, reference, current):
        if self._reference_model is None:
            self.fit_reference(reference, None)
        ref = self._standardize(reference)
        cur = self._standardize(current)
        repeats = self._effective_repeats(self.subsample_repeats, self.sample_budget,
                                          ref, cur)

        def draw():
            # Refit both sides per draw rather than only the current one: a
            # reference model frozen at a single subsample would leave that one
            # realisation in every score, and it is precisely such an asymmetry
            # between the offline and online pair that breaks exchangeability.
            # Warm-start each fit from the reference parameters -- two independent
            # EM runs land in different local optima, and on Forest that
            # optimisation noise is 20x larger than the drift signal, enough to
            # make the no-drift score exceed the drifted one.
            ref_model = (self._reference_model if repeats == 1
                         else self._fit_gmm(ref, warm_start=self._reference_model))
            cur_model = self._fit_gmm(cur, warm_start=ref_model)
            return _mc_js(ref_model, cur_model, self.mc_samples)

        return self._averaged(draw, repeats)


class GMMMarginalDetector(Detector):
    """Per-column 1-D Gaussian mixtures, Monte-Carlo JS, averaged over columns."""

    name = "gmm_marginal"

    def __init__(self, n_components: int = 3, sample_budget: Optional[int] = 50000,
                 mc_samples: int = 5000, reg_covar: float = 1e-4,
                 max_iter: int = 100, random_seed: int = 0,
                 subsample_repeats: int = DEFAULT_SUBSAMPLE_REPEATS):
        super().__init__(n_components=n_components, sample_budget=sample_budget,
                         mc_samples=mc_samples, reg_covar=reg_covar,
                         max_iter=max_iter, random_seed=random_seed,
                         subsample_repeats=subsample_repeats)
        self.n_components = n_components
        self.sample_budget = sample_budget
        self.mc_samples = mc_samples
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.subsample_repeats = subsample_repeats

    def _fit_column(self, column: np.ndarray, warm_start=None):
        from sklearn.mixture import GaussianMixture
        seed = self._next_seed()
        sample = self._subsample(column.reshape(-1, 1).astype(np.float64),
                                 self.sample_budget)
        n_components = max(1, min(self.n_components, np.unique(sample).size))
        init = {}
        if warm_start is not None and warm_start.n_components == n_components:
            init = dict(means_init=warm_start.means_,
                        weights_init=warm_start.weights_,
                        precisions_init=warm_start.precisions_)
        model = GaussianMixture(
            n_components=n_components, reg_covar=self.reg_covar,
            max_iter=self.max_iter, random_state=seed, **init,
        )
        return model.fit(sample)

    def fit_reference(self, reference, kinds):
        self._fit_scaler(reference)
        return self

    def distance(self, reference, current):
        ref = self._standardize(reference)
        cur = self._standardize(current)
        repeats = self._effective_repeats(self.subsample_repeats, self.sample_budget,
                                          ref, cur)
        per_column = []
        for j in range(ref.shape[1]):
            def draw(j=j):
                ref_model = self._fit_column(ref[:, j])
                # Warm start for the same reason as the joint variant: otherwise EM
                # local-optima noise dominates the drift signal.
                cur_model = self._fit_column(cur[:, j], warm_start=ref_model)
                return _mc_js(ref_model, cur_model, self.mc_samples)

            per_column.append(self._averaged(draw, repeats))
        return float(np.mean(per_column)) if per_column else 0.0


def _mc_js(model_p, model_q, mc_samples: int) -> float:
    """Monte-Carlo Jensen-Shannon divergence between two fitted mixtures, in [0, 1].

    Each mixture samples through its own ``random_state``, which ``_fit_gmm`` draws
    fresh per call, so the Monte-Carlo error averages out over repeated draws
    instead of staying frozen at one realisation.
    """
    xp, _ = model_p.sample(mc_samples)
    xq, _ = model_q.sample(mc_samples)

    def half(x, first, second):
        lf = first.score_samples(x)
        ls = second.score_samples(x)
        # log m = log((p + q) / 2) computed stably.
        lm = np.logaddexp(lf, ls) - np.log(2.0)
        return float(np.mean(lf - lm))

    js = 0.5 * (half(xp, model_p, model_q) + half(xq, model_q, model_p))
    return float(np.clip(js / np.log(2.0), 0.0, 1.0))


class MMDDetector(Detector):
    """Maximum mean discrepancy with an RBF kernel and the median heuristic.

    Quadratic in the subsample size, so the budget is what bounds the cost.
    """

    name = "mmd"

    def __init__(self, sample_budget: Optional[int] = 2000, random_seed: int = 0,
                 subsample_repeats: int = DEFAULT_SUBSAMPLE_REPEATS):
        super().__init__(sample_budget=sample_budget, random_seed=random_seed,
                         subsample_repeats=subsample_repeats)
        self.sample_budget = sample_budget
        self.random_seed = random_seed
        self.subsample_repeats = subsample_repeats

    def fit_reference(self, reference, kinds):
        self._fit_scaler(reference)
        return self

    def distance(self, reference, current):
        ref = np.asarray(reference)
        cur = np.asarray(current)
        repeats = self._effective_repeats(self.subsample_repeats, self.sample_budget,
                                          ref, cur)

        def sq_dists(x, y):
            d = ((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2.0 * x @ y.T)
            return np.maximum(d, 0.0)

        def draw():
            # Independent draws by construction now, so the old seed / seed + 1
            # offset is no longer needed to keep the two sides decorrelated.
            a = self._standardize(self._subsample(ref, self.sample_budget))
            b = self._standardize(self._subsample(cur, self.sample_budget))
            d_ab = sq_dists(a, b)
            median = np.median(d_ab)
            gamma = 1.0 / median if median > 0 else 1.0
            k = lambda d2: np.exp(-gamma * d2)
            mmd2 = (k(sq_dists(a, a)).mean() + k(sq_dists(b, b)).mean()
                    - 2.0 * k(d_ab).mean())
            return float(max(mmd2, 0.0))

        return self._averaged(draw, repeats)


class SlicedWassersteinDetector(Detector):
    """Squared sliced Wasserstein distance over random 1-D projections."""

    name = "sliced_wasserstein"

    def __init__(self, n_projections: int = 100, sample_budget: Optional[int] = 20000,
                 random_seed: int = 0,
                 subsample_repeats: int = DEFAULT_SUBSAMPLE_REPEATS):
        super().__init__(n_projections=n_projections, sample_budget=sample_budget,
                         random_seed=random_seed,
                         subsample_repeats=subsample_repeats)
        self.n_projections = n_projections
        self.sample_budget = sample_budget
        self.random_seed = random_seed
        self.subsample_repeats = subsample_repeats
        self._directions = None

    def fit_reference(self, reference, kinds):
        # Fixed directions and a fixed scale, so the statistic is comparable
        # across snapshots.
        self._fit_scaler(reference)
        rng = np.random.default_rng(self.random_seed)
        d = rng.normal(size=(np.asarray(reference).shape[1], self.n_projections))
        self._directions = d / np.linalg.norm(d, axis=0, keepdims=True)
        return self

    def distance(self, reference, current):
        if self._directions is None:
            self.fit_reference(reference, None)
        # Draw the same number of rows from each side: for equal sample sizes the
        # 1-D Wasserstein distance is just the mean squared difference of the
        # sorted projections, which avoids interpolating a quantile function per
        # projection.
        ref = np.asarray(reference)
        cur = np.asarray(current)
        n = min(ref.shape[0], cur.shape[0])
        if self.sample_budget is not None:
            n = min(n, int(self.sample_budget))
        # Equalising the two sides always truncates the larger snapshot, so what
        # binds here is n rather than sample_budget.
        repeats = self._effective_repeats(self.subsample_repeats, n, ref, cur)

        def draw():
            a = self._standardize(self._subsample(ref, n))
            b = self._standardize(self._subsample(cur, n))
            pa = np.sort(a @ self._directions, axis=0)
            pb = np.sort(b @ self._directions, axis=0)
            return float(np.mean((pa - pb) ** 2))

        return self._averaged(draw, repeats)


class FrequencySketchDetector(Detector):
    """Per-column equi-width histogram sketch compared by total variation.

    Stands in for the sketch-based detectors: very cheap, purely marginal, and
    blind to anything that leaves the per-column histograms alone.
    """

    name = "freq_sketch"

    def __init__(self, n_bins: int = 64):
        super().__init__(n_bins=n_bins)
        self.n_bins = n_bins
        self._edges: List[np.ndarray] = []

    def fit_reference(self, reference, kinds):
        ref = np.asarray(reference, dtype=np.float64)
        self._edges = []
        for j in range(ref.shape[1]):
            lo, hi = np.nanmin(ref[:, j]), np.nanmax(ref[:, j])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            self._edges.append(np.linspace(lo, hi, self.n_bins + 1)[1:-1])
        return self

    def distance(self, reference, current):
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(current, dtype=np.float64)
        if not self._edges:
            self.fit_reference(ref, None)
        per_column = []
        for j in range(ref.shape[1]):
            edges = self._edges[j]
            n_bins = len(edges) + 1
            hp = np.bincount(np.searchsorted(edges, ref[:, j], side="right"), minlength=n_bins)
            hq = np.bincount(np.searchsorted(edges, cur[:, j], side="right"), minlength=n_bins)
            p = hp / max(hp.sum(), 1)
            q = hq / max(hq.sum(), 1)
            per_column.append(0.5 * np.abs(p - q).sum())  # total variation
        return float(np.mean(per_column)) if per_column else 0.0


class PairwiseJointJSDetector(Detector):
    """Top-k pairwise JS without the IPF alignment.

    The ablation counterpart to :class:`DauceDetector`: it fires on marginal drift
    as well as dependency drift, so it cannot separate the two contributions.
    """

    name = "pairwise_joint_js"

    def __init__(self, n_bins: int = DEFAULT_N_BINS, top_k: Optional[int] = None):
        super().__init__(n_bins=n_bins, top_k=top_k)
        self.config = DistanceConfig(weight=1.0, n_bins=n_bins, top_k=top_k,
                                     align_marginals=False)
        self._distance = DriftDistance(self.config)

    def fit_reference(self, reference, kinds):
        self._distance.fit(reference, kinds)
        return self

    def distance(self, reference, current):
        return self._distance.evaluate(current).d_dependency


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
DETECTOR_REGISTRY = {
    "dauce": DauceDetector,
    "legacy_kl": LegacyGaussianKLDetector,
    "gmm_joint": GMMJointDetector,
    "gmm_marginal": GMMMarginalDetector,
    "mmd": MMDDetector,
    "sliced_wasserstein": SlicedWassersteinDetector,
    "freq_sketch": FrequencySketchDetector,
    "pairwise_joint_js": PairwiseJointJSDetector,
}


def build_detector(kind: str, **params) -> Detector:
    """Instantiate a detector by registry name."""
    if kind not in DETECTOR_REGISTRY:
        raise ValueError(f"unknown detector '{kind}'; "
                         f"available: {sorted(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[kind](**params)
