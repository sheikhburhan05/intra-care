"""Backward-compatible model exports and patient loader function."""

from typing import Optional

from src.domain.patient import LabReport, LabResult, Patient
from src.services.patient_data_loader import PatientDataLoader


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
