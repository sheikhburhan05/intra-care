"""Utility functions for CSV/Excel logging and JSON serialization."""
import json
import os
import glob
from typing import Dict, Any

import pandas as pd


def log_csv_headers(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Read CSVs, log headers and sample rows, return {filename: df}."""
    csv_files = [p for p in glob.glob(os.path.join(base_dir, "*.csv")) if "venv" not in p]
    results = {}
    for path in csv_files:
        name = os.path.basename(path)
        df = pd.read_csv(path)
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print("HEADER:", df.columns.tolist())
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
        print("First 5 rows:")
        print(df.head(5).to_string())
        results[name] = df
    return results


def log_xlsx_headers(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    """Read Excel sheets, log headers and sample rows."""
    if not os.path.exists(xlsx_path):
        return {}
    xl = pd.ExcelFile(xlsx_path)
    results = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        print(f"\n{'='*60}\n{os.path.basename(xlsx_path)} (sheet: {sheet})\n{'='*60}")
        print("HEADER:", df.columns.tolist())
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
        print("First 5 rows:")
        print(df.head(5).to_string())
        results[sheet] = df
    return results


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas types and NaN to JSON-serializable values."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, (int, float)) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def patients_to_json_serializable(patients: Any) -> dict:
    """Convert Patient objects to JSON-serializable dict."""
    out = {}
    for pid, p in patients.items():
        lab_reports = [
            {
                "lab_report_id": lr.lab_report_id,
                "results": [{"labanalyte": r.labanalyte, "labvalue": r.labvalue} for r in lr.results],
            }
            for lr in p.lab_reports
        ]
        out[str(pid)] = {
            "patient_id": pid,
            "lab_reports": lab_reports,
            "medications": _sanitize_for_json(p.medications),
            "problems": _sanitize_for_json(p.problems),
        }
    return out


def save_patients_to_json(patients: Any, json_path: str) -> None:
    """Save Patient objects to a JSON file."""
    data = patients_to_json_serializable(patients)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def load_patients_from_json(json_path: str) -> Dict[str, Any]:
    """Load patient data from a JSON file into a dict."""
    with open(json_path) as f:
        return json.load(f)


def get_enterpriseid_column(df: pd.DataFrame) -> str:
    """Return the enterpriseid column name from a dataframe."""
    return next(
        (c for c in df.columns if str(c).strip().lower() == "enterpriseid"),
        df.columns[0],
    )
