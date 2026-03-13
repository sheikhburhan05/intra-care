"""Main entry point. Loads patient data from JSON file into a dict."""
import os

from utils import load_patients_from_json

TARGET_PATIENT_ID = "123455"

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base, "patients.json")

    patients = load_patients_from_json(json_path)

    print(f"Loaded {len(patients)} patients from {json_path}\n")
    for pid, p in list(patients.items())[:2]:
        lab_count = len(p.get("lab_reports", []))
        med_count = len(p.get("medications", []))
        prob_count = len(p.get("problems", []))
        print(f"Patient {pid} (enterpriseid): {lab_count} lab reports, {med_count} meds, {prob_count} problems")
    print()

    if TARGET_PATIENT_ID in patients:
        p = patients[TARGET_PATIENT_ID]
        print(f"\n{'='*60}\nPatient ID (enterpriseid): {TARGET_PATIENT_ID}\n{'='*60}")
        print(f"Lab reports: {len(p.get('lab_reports', []))}")
        for i, lr in enumerate(p.get("lab_reports", [])[:3]):
            print(f"  LabReport[{i+1}] id={lr.get('lab_report_id', '')}")
            for r in lr.get("results", [])[:2]:
                print(f"    {r.get('labanalyte')}: {r.get('labvalue')}")
        print(f"\nMedications: {p.get('medications', [])}")
        print(f"Problems: {p.get('problems', [])}")
    else:
        print(f"\nPatient ID {TARGET_PATIENT_ID} not found.")
