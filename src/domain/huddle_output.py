"""Pydantic schema for structured huddle analysis output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MedicationGap(BaseModel):
    medication: str = Field(description="The medication that implies a condition")
    implied_condition: str = Field(description="ICD-10/clinical condition the medication suggests")
    evidence: str = Field(description="Why this medication implies the condition")
    in_problem_list: bool = Field(description="Whether this condition is in the patient's ICD-10 problem list")


class LabGap(BaseModel):
    lab_analyte: str = Field(description="Lab test name")
    lab_value: str = Field(description="Reported value")
    implied_condition: str = Field(description="Condition suggested by abnormal lab")
    evidence: str = Field(description="Clinical threshold or reasoning")
    in_problem_list: bool = Field(description="Whether this condition is in the patient's ICD-10 problem list")


class MedicationToDiagnosis(BaseModel):
    suspected_gaps: list[MedicationGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of medication-diagnosis gaps")


class LabReportToDiagnosis(BaseModel):
    suspected_gaps: list[LabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of lab-diagnosis gaps")


class SummaryNoteBeforeHuddle(BaseModel):
    context: str = Field(description="Medication, lab, prior note, or diagnosis context supporting the flags")
    suggested_huddle_note_bullet: str = Field(description="Suggested huddle note bullet point")
    physician_prompt: str = Field(description="Short physician-facing discussion prompt or coding reminder")


class HuddleAnalysisOutput(BaseModel):
    medication_to_diagnosis: MedicationToDiagnosis
    lab_report_to_diagnosis: LabReportToDiagnosis
    summary_note_before_huddle: SummaryNoteBeforeHuddle
