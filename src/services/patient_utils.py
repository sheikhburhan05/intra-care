"""Helpers for extracting and formatting patient data from raw JSON records."""

from __future__ import annotations

import re
from typing import Any


def extract_medications(patient: dict[str, Any]) -> list[str]:
    """Return a deduplicated flat list of medication name strings for the patient."""
    medications: list[str] = []
    for medication in patient.get("medications", []):
        names = medication.get("med names") or medication.get("med_names") or ""
        for part in re.split(r"[;,]+\s*", str(names)):
            part = part.strip()
            if part:
                medications.append(part)
    return list(dict.fromkeys(medications))


def extract_problems(patient: dict[str, Any]) -> list[str]:
    """Return a list of problem description strings from the patient's problem list."""
    return [
        (problem.get("patientsnomedproblemdesc") or problem.get("problem_desc") or "").strip()
        for problem in patient.get("problems", [])
        if (problem.get("patientsnomedproblemdesc") or problem.get("problem_desc"))
    ]


def extract_labs(patient: dict[str, Any]) -> list[dict[str, str]]:
    """Return deduplicated (analyte, value) pairs across all lab reports."""
    seen: set[tuple[str, str]] = set()
    labs: list[dict[str, str]] = []
    for report in patient.get("lab_reports", []):
        for result in report.get("results", []):
            analyte = (result.get("labanalyte") or "").strip()
            value = (result.get("labvalue") or "").strip()
            if analyte and (analyte, value) not in seen:
                seen.add((analyte, value))
                labs.append({"labanalyte": analyte, "labvalue": value})
    return labs


def format_report_results(report: dict[str, Any]) -> str:
    """Render a single lab report's results as a bullet list for prompt injection."""
    lines: list[str] = []
    for result in report.get("results", []):
        analyte = str(result.get("labanalyte", "")).strip()
        value = str(result.get("labvalue", "")).strip()
        if analyte:
            lines.append(f"- {analyte}: {value}")
    return "\n".join(lines)


def format_all_reports_for_combined(lab_reports: list[dict[str, Any]]) -> str:
    """Render a snapshot of all lab reports for the combined multi-report prompt."""
    chunks: list[str] = []
    for report in lab_reports:
        report_id = str(report.get("lab_report_id", "")).strip() or "unknown"
        lines = [f"Report ID: {report_id}"]
        for result in report.get("results", [])[:40]:
            analyte = str(result.get("labanalyte", "")).strip()
            value = str(result.get("labvalue", "")).strip()
            if analyte:
                lines.append(f"- {analyte}: {value}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks[:50])


def short_report_name(report_id: str, max_len: int = 40) -> str:
    """Return a concise display name from a raw lab report ID string.

    IDs often look like '005009    CBC WITH DIFFERENTIAL/PLATELET'.
    We strip the leading numeric code and return only the descriptive part.
    """
    parts = re.split(r"\s{2,}", report_id.strip(), maxsplit=1)
    name = parts[-1] if len(parts) > 1 else report_id
    return name[:max_len] + ("…" if len(name) > max_len else "")
