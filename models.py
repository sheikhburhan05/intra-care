"""Backward-compatible model exports and patient loader function."""

from pathlib import Path
import sys
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poc.domain.patient import LabReport, LabResult, Patient
from poc.services.patient_data_loader import PatientDataLoader


def load_patients(
    lab_csv_path: str,
    medication_xlsx_path: Optional[str] = None,
    appointment_csv_path: Optional[str] = None,
) -> dict[int, Patient]:
    return PatientDataLoader().load_patients(
        lab_csv_path=lab_csv_path,
        medication_xlsx_path=medication_xlsx_path,
        appointment_csv_path=appointment_csv_path,
    )


__all__ = ["LabResult", "LabReport", "Patient", "load_patients"]
