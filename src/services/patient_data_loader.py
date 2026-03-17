"""Service for assembling patient objects from source files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..domain.patient import LabReport, LabResult, Patient
from ..repositories.patient_repository import PatientRepository


def _safe_str(val: object) -> str:
    """Convert value to str, treating NaN/None as empty string (JSON-safe)."""
    if pd.isna(val):
        return ""
    return str(val) if val is not None else ""


class PatientDataLoader:
    """Load patient entities from lab orders and optional medication workbook."""

    def __init__(self, repository: Optional[PatientRepository] = None) -> None:
        self.repository = repository or PatientRepository()

    def load_patients(
        self,
        lab_csv_path: str,
        medication_xlsx_path: Optional[str] = None,
        appointment_csv_path: Optional[str] = None,  # reserved for future use
    ) -> dict[int, Patient]:
        del appointment_csv_path
        lab_df = pd.read_csv(lab_csv_path)
        cols = list(lab_df.columns)
        ent_col = next((c for c in cols if c.lower() == "enterpriseid"), cols[0])
        lab_dtl_col = next((c for c in cols if "lab" in c.lower() and "dtl" in c.lower()), cols[2])
        analyte_col = next((c for c in cols if "analyte" in c.lower()), cols[3])
        value_col = next(
            (c for c in cols if "labvalue" in c.lower() or (c != analyte_col and "value" in c.lower())),
            cols[4],
        )

        patients: dict[int, Patient] = {}
        for (enterprise_id, lab_detail), group in lab_df.groupby([ent_col, lab_dtl_col]):
            enterprise_id = int(enterprise_id)
            if enterprise_id not in patients:
                patients[enterprise_id] = Patient(patient_id=enterprise_id)
            results = [
                LabResult(
                    _safe_str(row[analyte_col]),
                    _safe_str(row[value_col]),
                )
                for _, row in group.iterrows()
            ]
            patients[enterprise_id].lab_reports.append(
                LabReport(lab_report_id=str(lab_detail).strip(), results=results)
            )

        if medication_xlsx_path and Path(medication_xlsx_path).exists():
            self._attach_medications_and_problems(patients, medication_xlsx_path)

        return patients

    def _attach_medications_and_problems(self, patients: dict[int, Patient], medication_xlsx_path: str) -> None:
        workbook = pd.ExcelFile(medication_xlsx_path)
        medication_sheet = (
            "Medications"
            if "Medications" in workbook.sheet_names
            else ("Medication" if "Medication" in workbook.sheet_names else None)
        )
        if medication_sheet:
            medication_df = pd.read_excel(medication_xlsx_path, sheet_name=medication_sheet)
            enterprise_column = self.repository.get_enterpriseid_column(medication_df)
            for enterprise_id, group in medication_df.groupby(enterprise_column):
                if str(enterprise_id).strip().lower() == "enterpriseid":
                    continue
                self._get_or_create_patient(patients, int(enterprise_id)).medications = group.to_dict("records")

        if "Problem List" not in workbook.sheet_names:
            return
        problem_df = pd.read_excel(medication_xlsx_path, sheet_name="Problem List")
        if "Unnamed" in str(problem_df.columns[0]):
            problem_df.columns = problem_df.iloc[0]
            problem_df = problem_df.drop(0).reset_index(drop=True)
        enterprise_column = self.repository.get_enterpriseid_column(problem_df)
        for enterprise_id, group in problem_df.groupby(enterprise_column):
            if str(enterprise_id).strip().lower() == "enterpriseid":
                continue
            self._get_or_create_patient(patients, int(enterprise_id)).problems = group.to_dict("records")

    @staticmethod
    def _get_or_create_patient(patients: dict[int, Patient], enterprise_id: int) -> Patient:
        if enterprise_id not in patients:
            patients[enterprise_id] = Patient(patient_id=enterprise_id)
        return patients[enterprise_id]
