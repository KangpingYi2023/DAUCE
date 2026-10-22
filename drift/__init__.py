"""Data-driven drift detection for learned cardinality estimators.

The detector is model-agnostic: it reads table statistics only and never calls a
CE model, computes a loss, or touches the query workload. Nothing in this package
imports torch.

Typical use::

    from drift import data, detectors

    table, kinds = data.load_dataset("census")
    detector = detectors.build_detector("dauce")
    detector.fit_reference(table, kinds)
    detector.calibrate(table, kinds, batch_size=1600, tau=0.05)
    result = detector.detect(table, updated_table)
"""

from drift.calibration import CalibrationResult, calibrate
from drift.distance import (DistanceBreakdown, DistanceConfig, DriftDistance,
                            drift_distance, legacy_column_kl)
from drift.detectors import (DETECTOR_REGISTRY, DetectionResult, Detector,
                             build_detector)
from drift.scenarios import Scenario, expand_scenarios
from drift.schema import ColumnKind, infer_kinds

__all__ = [
    "CalibrationResult",
    "calibrate",
    "ColumnKind",
    "infer_kinds",
    "DistanceBreakdown",
    "DistanceConfig",
    "DriftDistance",
    "drift_distance",
    "legacy_column_kl",
    "DETECTOR_REGISTRY",
    "DetectionResult",
    "Detector",
    "build_detector",
    "Scenario",
    "expand_scenarios",
]
