"""Persistence helpers for patient JSON data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..domain.patient import Patient


class PatientRepository:
    """Read/write patient payloads and convert domain models into JSON shape."""

    @staticmethod
    def load_patients_from_json(json_path: str | Path) -> dict[str, Any]:
        with Path(json_path).open(encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def save_patients_to_json(cls, patients: dict[int, Patient], json_path: str | Path) -> None:
        payload = cls.patients_to_json_serializable(patients)
        with Path(json_path).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def patients_to_json_serializable(cls, patients: dict[int, Patient]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for patient_id, patient in patients.items():
            lab_reports = [
                {
                    "lab_report_id": report.lab_report_id,
                    "results": [{"labanalyte": r.labanalyte, "labvalue": r.labvalue} for r in report.results],
                }
                for report in patient.lab_reports
            ]
            output[str(patient_id)] = {
                "patient_id": patient_id,
                "lab_reports": lab_reports,
                "medications": cls._sanitize_for_json(patient.medications),
                "problems": cls._sanitize_for_json(patient.problems),
            }
        return output

    @staticmethod
    def get_enterpriseid_column(df: pd.DataFrame) -> str:
        return next((c for c in df.columns if str(c).strip().lower() == "enterpriseid"), df.columns[0])

    @classmethod
    def _sanitize_for_json(cls, obj: Any) -> Any:
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if isinstance(obj, (int, float)) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        if isinstance(obj, dict):
            return {str(k): cls._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitize_for_json(item) for item in obj]
        return obj
