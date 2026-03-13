"""Data models and load logic."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from utils import get_enterpriseid_column


@dataclass
class LabResult:
    labanalyte: str
    labvalue: str


@dataclass
class LabReport:
    lab_report_id: str  # lab ord dtl
    results: List[LabResult]


@dataclass
class Patient:
    """patient_id = enterpriseid. lab_reports, medications, problems directly under patient."""
    patient_id: int
    lab_reports: List[LabReport] = field(default_factory=list)
    medications: List[Dict] = field(default_factory=list)
    problems: List[Dict] = field(default_factory=list)


def load_patients(
    lab_csv_path: str,
    medication_xlsx_path: Optional[str] = None,
    appointment_csv_path: Optional[str] = None,  # reserved for future use
) -> Dict[int, Patient]:
    """Load patients. patient_id = enterpriseid. lab_reports, medications, problems directly under patient."""
    lab_df = pd.read_csv(lab_csv_path)
    cols = list(lab_df.columns)
    ent_col = next((c for c in cols if c.lower() == "enterpriseid"), cols[0])
    lab_dtl_col = next((c for c in cols if "lab" in c.lower() and "dtl" in c.lower()), cols[2])
    analyte_col = next((c for c in cols if "analyte" in c.lower()), cols[3])
    value_col = next(
        (c for c in cols if "labvalue" in c.lower() or (c != analyte_col and "value" in c.lower())),
        cols[4],
    )

    patients: Dict[int, Patient] = {}

    for (eid, lab_dtl), grp in lab_df.groupby([ent_col, lab_dtl_col]):
        eid = int(eid)
        if eid not in patients:
            patients[eid] = Patient(patient_id=eid)
        results = [LabResult(row[analyte_col], str(row[value_col])) for _, row in grp.iterrows()]
        patients[eid].lab_reports.append(LabReport(lab_report_id=str(lab_dtl).strip(), results=results))

    if medication_xlsx_path and os.path.exists(medication_xlsx_path):
        xl = pd.ExcelFile(medication_xlsx_path)
        med_sheet = (
            "Medications"
            if "Medications" in xl.sheet_names
            else ("Medication" if "Medication" in xl.sheet_names else None)
        )
        if med_sheet:
            med_df = pd.read_excel(medication_xlsx_path, sheet_name=med_sheet)
            ent = get_enterpriseid_column(med_df)
            for eid, grp in med_df.groupby(ent):
                if str(eid).strip().lower() == "enterpriseid":
                    continue
                eid = int(eid)
                if eid not in patients:
                    patients[eid] = Patient(patient_id=eid)
                patients[eid].medications = grp.to_dict("records")
        if "Problem List" in xl.sheet_names:
            prob_df = pd.read_excel(medication_xlsx_path, sheet_name="Problem List")
            if "Unnamed" in str(prob_df.columns[0]):
                prob_df.columns = prob_df.iloc[0]
                prob_df = prob_df.drop(0).reset_index(drop=True)
            ent = get_enterpriseid_column(prob_df)
            for eid, grp in prob_df.groupby(ent):
                if str(eid).strip().lower() == "enterpriseid":
                    continue
                eid = int(eid)
                if eid not in patients:
                    patients[eid] = Patient(patient_id=eid)
                patients[eid].problems = grp.to_dict("records")

    return patients
