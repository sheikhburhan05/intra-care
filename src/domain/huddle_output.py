"""Pydantic schema for structured huddle analysis output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MedicationGap(BaseModel):
    medication: str = Field(description="The medication that implies a condition")
    implied_condition: str = Field(description="ICD-10/clinical condition the medication suggests")
    icd10_code: str = Field(default="UNKNOWN", description="Best matching ICD-10 code for the implied condition")
    evidence: str = Field(
        description=(
            "Detailed rationale including medication-to-indication mapping and explicit mismatch with current "
            "problem list (or why no clear linked diagnosis exists)."
        )
    )
    in_problem_list: bool = Field(description="Whether this condition is in the patient's ICD-10 problem list")


class LabGap(BaseModel):
    lab_report_id: str = Field(default="", description="Lab report ID where this gap was detected")
    lab_analyte: str = Field(description="Lab test name")
    lab_value: str = Field(description="Reported value")
    implied_condition: str = Field(description="Condition suggested by abnormal lab")
    icd10_code: str = Field(default="UNKNOWN", description="Best matching ICD-10 code for the implied condition")
    evidence: str = Field(description="Clinical threshold or reasoning")
    in_problem_list: bool = Field(description="Whether this condition is in the patient's ICD-10 problem list")


class MedicationToDiagnosis(BaseModel):
    suspected_gaps: list[MedicationGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of medication-diagnosis gaps")


class LabReportToDiagnosis(BaseModel):
    suspected_gaps: list[LabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of lab-diagnosis gaps")


class CombinedLabGap(BaseModel):
    implied_condition: str = Field(description="Condition suggested by combined multi-report evidence")
    icd10_code: str = Field(default="UNKNOWN", description="Best matching ICD-10 code for the implied condition")
    evidence: str = Field(description="Cross-report rationale or threshold-based trend")
    contributing_report_ids: list[str] = Field(default_factory=list, description="Report IDs supporting the finding")
    in_problem_list: bool = Field(description="Whether this condition is already in the patient's problem list")


class CombinedLabReportToDiagnosis(BaseModel):
    suspected_gaps: list[CombinedLabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of combined multi-report diagnosis gaps")


class SummaryNoteBeforeHuddle(BaseModel):
    context: str = Field(description="Medication, lab, prior note, or diagnosis context supporting the flags")
    suggested_huddle_note_bullet: str = Field(description="Suggested huddle note bullet point")
    physician_prompt: str = Field(description="Short physician-facing discussion prompt or coding reminder")


class HuddleAnalysisOutput(BaseModel):
    medication_to_diagnosis: MedicationToDiagnosis
    lab_report_to_diagnosis: LabReportToDiagnosis
    combined_lab_report_to_diagnosis: CombinedLabReportToDiagnosis
    summary_note_before_huddle: SummaryNoteBeforeHuddle
