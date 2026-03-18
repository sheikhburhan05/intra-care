#!/usr/bin/env python3
"""Interactive CLI for running huddle analysis on one or more patients.

Usage
-----
    python run_huddle.py
    # or with the venv:
    venv/bin/python3 run_huddle.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from src.config import output_dir
from src.services.huddle_analyzer import HuddleAnalyzer

# ── Menu definitions ──────────────────────────────────────────────────────────

ANALYSES = [
    ("medication",  "Medication Gap Analysis"),
    ("per_report",  "Per-Report Lab Analysis"),
    ("multi_report","Multi-Report (Combined) Analysis"),
    ("summary",     "Doctor Pre-Huddle Summary"),
]

SEP = "=" * 58


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ids(raw: str) -> list[str]:
    return [p for p in re.split(r"[\s,;]+", raw.strip()) if p]


def _ask_patient_ids() -> list[str]:
    print(f"\n{SEP}")
    print("  Huddle Analysis Runner")
    print(SEP)
    print("Enter one or more patient IDs (space or comma separated).\n")
    while True:
        raw = input("Patient IDs: ").strip()
        ids = _parse_ids(raw)
        if ids:
            return ids
        print("  [!] Please enter at least one patient ID.\n")


def _ask_analyses() -> dict[str, bool]:
    print(f"\n{SEP}")
    print("  Select analyses to run")
    print(SEP)
    for i, (_, label) in enumerate(ANALYSES, start=1):
        print(f"  {i}. {label}")
    print()
    print("Enter numbers separated by spaces/commas, or press Enter to run ALL.")
    print("Example: 1 3  →  runs Medication + Multi-Report only\n")

    while True:
        raw = input("Your choice: ").strip()

        if not raw:
            return {key: True for key, _ in ANALYSES}

        tokens = re.split(r"[\s,;]+", raw)
        chosen: set[int] = set()
        valid = True
        for token in tokens:
            if token.isdigit() and 1 <= int(token) <= len(ANALYSES):
                chosen.add(int(token))
            else:
                print(f"  [!] '{token}' is not a valid option. Choose 1–{len(ANALYSES)}.\n")
                valid = False
                break

        if valid and chosen:
            selected = {key: (i in chosen) for i, (key, _) in enumerate(ANALYSES, start=1)}
            return selected


def _print_selection(patient_ids: list[str], selected: dict[str, bool]) -> None:
    print(f"\n{SEP}")
    print(f"  Patients : {', '.join(patient_ids)}")
    print("  Analyses :")
    for key, label in ANALYSES:
        tick = "✓" if selected.get(key) else "✗"
        print(f"    [{tick}] {label}")
    print(SEP)


def _confirm() -> bool:
    ans = input("\nProceed? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    patient_ids = _ask_patient_ids()
    selected = _ask_analyses()
    _print_selection(patient_ids, selected)

    if not _confirm():
        print("Aborted.")
        sys.exit(0)

    analyzer = HuddleAnalyzer()

    for pid in patient_ids:
        print(f"\n{SEP}")
        print(f"  Running analysis for patient: {pid}")
        print(SEP)
        try:
            analyzer.analyze_patient_huddle(
                patient_id=pid,
                output_dir=str(output_dir()),
                enable_medication_analysis=selected["medication"],
                enable_per_report_lab_analysis=selected["per_report"],
                enable_combined_lab_analysis=selected["multi_report"],
                enable_doctor_summary=selected["summary"],
            )
        except ValueError as exc:
            print(f"  [error] {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] Unexpected error for patient {pid}: {exc}")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
