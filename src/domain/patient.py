"""Domain models for patient data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LabResult:
    labanalyte: str
    labvalue: str


@dataclass
class LabReport:
    lab_report_id: str
    results: list[LabResult]


@dataclass
class Patient:
    """Patient entity keyed by enterprise ID."""

    patient_id: int
    lab_reports: list[LabReport] = field(default_factory=list)
    medications: list[dict[str, Any]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)
