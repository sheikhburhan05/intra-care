"""Helpers for deduplicating gap lists and building fallback summary strings."""

from __future__ import annotations

from typing import Any


def dedupe_lab_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate per-report lab gaps keyed on (report_id, analyte, value, condition)."""
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for gap in gaps:
        key = (
            str(gap.get("lab_report_id", "")).strip().lower(),
            str(gap.get("lab_analyte", "")).strip().lower(),
            str(gap.get("lab_value", "")).strip().lower(),
            str(gap.get("implied_condition", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gap)
    return deduped


def dedupe_combined_lab_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate combined-lab gaps keyed on (condition, sorted contributing report IDs)."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[dict[str, Any]] = []
    for gap in gaps:
        report_ids = tuple(sorted(set(gap.get("contributing_report_ids", []))))
        key = (str(gap.get("implied_condition", "")).strip().lower(), report_ids)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gap)
    return deduped


def build_medication_summary(medication_gaps: list[dict[str, Any]]) -> str:
    """Return a plain-text fallback summary for medication gaps."""
    if not medication_gaps:
        return "No medication-diagnosis gaps identified."
    parts = [
        f"{gap.get('medication', 'Unknown')} implies "
        f"{gap.get('implied_condition', 'unknown condition')} "
        f"({gap.get('icd10_code', 'UNKNOWN')}) not present in problem list"
        for gap in medication_gaps
    ]
    return "; ".join(parts) + "."


def build_combined_lab_summary(combined_gaps: list[dict[str, Any]]) -> str:
    """Return a plain-text fallback summary for combined-lab gaps."""
    if not combined_gaps:
        return "No additional combined multi-report diagnosis gaps identified."
    return f"Detected {len(combined_gaps)} combined multi-report diagnosis gap(s)."
