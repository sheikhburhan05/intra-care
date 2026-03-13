#!/usr/bin/env python3
"""CLI and backward-compatible exports for huddle analysis."""

from pathlib import Path
import argparse
import sys
from typing import Any, Optional

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import DEFAULT_HUDDLE_MODEL
from src.domain.huddle_output import (
    HuddleAnalysisOutput,
    LabGap,
    LabReportToDiagnosis,
    MedicationGap,
    MedicationToDiagnosis,
    SummaryNoteBeforeHuddle,
)
from src.services.huddle_analyzer import HuddleAnalyzer


def analyze_patient_huddle(
    patient_id: str,
    patients_json_path: str = "patients.json",
    output_dir: Optional[str] = None,
    model: str = DEFAULT_HUDDLE_MODEL,
    use_web_search: bool = True,
    use_llm_tools: bool = True,
) -> dict[str, Any]:
    """Compatibility function for callers expecting module-level API."""
    return HuddleAnalyzer().analyze_patient_huddle(
        patient_id=patient_id,
        patients_json_path=patients_json_path,
        output_dir=output_dir,
        model=model,
        use_web_search=use_web_search,
        use_llm_tools=use_llm_tools,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze patient for medication/lab-diagnosis gaps")
    parser.add_argument("patient_id", help="Patient ID (enterpriseid)")
    parser.add_argument("--patients-json", default="patients.json", help="Path to patients.json")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for {patientId}.json")
    parser.add_argument("--model", default=DEFAULT_HUDDLE_MODEL, help="Claude model name")
    parser.add_argument("--no-search", action="store_true", help="Skip web search for thresholds")
    parser.add_argument("--no-llm-tools", action="store_true", help="Use fixed search queries")
    args = parser.parse_args()

    analyze_patient_huddle(
        patient_id=args.patient_id,
        patients_json_path=args.patients_json,
        output_dir=args.output_dir,
        model=args.model,
        use_web_search=not args.no_search,
        use_llm_tools=not args.no_llm_tools,
    )


__all__ = [
    "MedicationGap",
    "LabGap",
    "MedicationToDiagnosis",
    "LabReportToDiagnosis",
    "SummaryNoteBeforeHuddle",
    "HuddleAnalysisOutput",
    "analyze_patient_huddle",
]
