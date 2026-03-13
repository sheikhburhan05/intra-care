"""Quick local inspection entrypoint for patients.json."""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poc.config import resolve_repo_path
from poc.repositories.patient_repository import PatientRepository

TARGET_PATIENT_ID = "123455"


def main() -> None:
    json_path = resolve_repo_path("patients.json")
    patients = PatientRepository.load_patients_from_json(json_path)

    print(f"Loaded {len(patients)} patients from {json_path}\n")
    for patient_id, patient in list(patients.items())[:2]:
        lab_count = len(patient.get("lab_reports", []))
        med_count = len(patient.get("medications", []))
        problem_count = len(patient.get("problems", []))
        print(
            f"Patient {patient_id} (enterpriseid): "
            f"{lab_count} lab reports, {med_count} meds, {problem_count} problems"
        )
    print()

    if TARGET_PATIENT_ID not in patients:
        print(f"\nPatient ID {TARGET_PATIENT_ID} not found.")
        return

    patient = patients[TARGET_PATIENT_ID]
    print(f"\n{'=' * 60}\nPatient ID (enterpriseid): {TARGET_PATIENT_ID}\n{'=' * 60}")
    print(f"Lab reports: {len(patient.get('lab_reports', []))}")
    for index, report in enumerate(patient.get("lab_reports", [])[:3], start=1):
        print(f"  LabReport[{index}] id={report.get('lab_report_id', '')}")
        for result in report.get("results", [])[:2]:
            print(f"    {result.get('labanalyte')}: {result.get('labvalue')}")
    print(f"\nMedications: {patient.get('medications', [])}")
    print(f"Problems: {patient.get('problems', [])}")


if __name__ == "__main__":
    main()
