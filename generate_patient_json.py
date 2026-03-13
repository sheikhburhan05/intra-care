#!/usr/bin/env python3
"""Generate patients.json from lab orders and medication sources."""

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poc.config import resolve_repo_path
from poc.repositories.patient_repository import PatientRepository
from poc.services.patient_data_loader import PatientDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patient data JSON from CSV and Excel")
    parser.add_argument(
        "--lab-csv",
        default="data/Lab orders.csv",
        help="Path to Lab orders CSV (default: data/Lab orders.csv)",
    )
    parser.add_argument(
        "--medication-xlsx",
        default="data/Medication.xlsx",
        help="Path to Medication.xlsx (default: data/Medication.xlsx)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="patients.json",
        help="Output JSON file path (default: patients.json)",
    )
    args = parser.parse_args()

    lab_path = resolve_repo_path(args.lab_csv)
    med_path = resolve_repo_path(args.medication_xlsx)
    output_path = resolve_repo_path(args.output)

    loader = PatientDataLoader()
    repository = PatientRepository()
    patients = loader.load_patients(
        lab_csv_path=str(lab_path),
        medication_xlsx_path=str(med_path) if med_path.exists() else None,
    )
    repository.save_patients_to_json(patients, output_path)
    print(f"Generated {output_path} with {len(patients)} patients")


if __name__ == "__main__":
    main()
