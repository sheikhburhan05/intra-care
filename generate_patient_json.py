#!/usr/bin/env python3
"""Generate patients.json from Lab orders CSV and Medication.xlsx."""
import argparse
import os

from models import load_patients
from utils import save_patients_to_json


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

    base = os.path.dirname(os.path.abspath(__file__))
    lab_path = args.lab_csv if os.path.isabs(args.lab_csv) else os.path.join(base, args.lab_csv)
    med_path = args.medication_xlsx if os.path.isabs(args.medication_xlsx) else os.path.join(base, args.medication_xlsx)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)

    patients = load_patients(
        lab_csv_path=lab_path,
        medication_xlsx_path=med_path if os.path.exists(med_path) else None,
    )
    save_patients_to_json(patients, out_path)
    print(f"Generated {out_path} with {len(patients)} patients")


if __name__ == "__main__":
    main()
