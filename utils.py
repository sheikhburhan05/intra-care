"""Utility helpers for local data inspection and compatibility APIs."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.repositories.patient_repository import PatientRepository


def log_csv_headers(base_dir: str) -> dict[str, pd.DataFrame]:
    """Read CSV files, print headers/sample rows, and return loaded dataframes."""
    csv_files = [path for path in glob.glob(os.path.join(base_dir, "*.csv")) if "venv" not in path]
    results: dict[str, pd.DataFrame] = {}
    for path in csv_files:
        name = os.path.basename(path)
        dataframe = pd.read_csv(path)
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print("HEADER:", dataframe.columns.tolist())
        print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} cols")
        print("First 5 rows:")
        print(dataframe.head(5).to_string())
        results[name] = dataframe
    return results


def log_xlsx_headers(xlsx_path: str) -> dict[str, pd.DataFrame]:
    """Read workbook sheets, print headers/sample rows, and return loaded dataframes."""
    if not os.path.exists(xlsx_path):
        return {}
    workbook = pd.ExcelFile(xlsx_path)
    results: dict[str, pd.DataFrame] = {}
    for sheet in workbook.sheet_names:
        dataframe = pd.read_excel(xlsx_path, sheet_name=sheet)
        print(f"\n{'=' * 60}\n{os.path.basename(xlsx_path)} (sheet: {sheet})\n{'=' * 60}")
        print("HEADER:", dataframe.columns.tolist())
        print(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} cols")
        print("First 5 rows:")
        print(dataframe.head(5).to_string())
        results[sheet] = dataframe
    return results


def patients_to_json_serializable(patients: Any) -> dict[str, Any]:
    return PatientRepository.patients_to_json_serializable(patients)


def save_patients_to_json(patients: Any, json_path: str | Path) -> None:
    PatientRepository.save_patients_to_json(patients, json_path)


def load_patients_from_json(json_path: str | Path) -> dict[str, Any]:
    return PatientRepository.load_patients_from_json(json_path)


def get_enterpriseid_column(df: pd.DataFrame) -> str:
    return PatientRepository.get_enterpriseid_column(df)
