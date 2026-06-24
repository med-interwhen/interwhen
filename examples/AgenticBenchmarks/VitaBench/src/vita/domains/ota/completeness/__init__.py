"""
OTA Completeness checking.

Two parts:
  - Extraction (offline): derive minimal booking-count constraints from a task.
  - Checking (runtime): verify the final order state satisfies those constraints.

Usage:
    from vita.domains.ota.completeness import check_completeness, CompletenessConstraints
    from vita.domains.ota.completeness import extract_completeness_constraints
"""

from vita.domains.ota.completeness.schema import (
    AttractionCompleteness,
    CancelCompleteness,
    CompletenessConstraints,
    FlightCompleteness,
    HotelCompleteness,
    ModifyCompleteness,
    TrainCompleteness,
)
from vita.domains.ota.completeness.checker import check_completeness
from vita.domains.ota.completeness.constraint_extractor import extract_completeness_constraints

__all__ = [
    "check_completeness",
    "extract_completeness_constraints",
    "CompletenessConstraints",
    "HotelCompleteness",
    "FlightCompleteness",
    "TrainCompleteness",
    "AttractionCompleteness",
    "CancelCompleteness",
    "ModifyCompleteness",
]
