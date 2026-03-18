"""CLI script: export huddle analysis JSON files to data.xlsx.

Usage
-----
Run from the project root (same folder that contains the patient JSON files):

    python export_to_excel.py

You will be prompted to enter one or more patient IDs, e.g.:

    123455 123459 123460
    # or comma-separated
    123455, 123459, 123460
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Load the exporter directly to avoid triggering the full package __init__
# (which imports heavy LLM dependencies not needed here).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "huddle_excel_exporter",
    Path(__file__).parent / "src" / "services" / "huddle_excel_exporter.py",
)
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
HuddleExcelExporter = _mod.HuddleExcelExporter


def _parse_ids(raw: str) -> list[str]:
    """Split a free-form string of patient IDs (space / comma / newline separated)."""
    return [p for p in re.split(r"[\s,;]+", raw.strip()) if p]


def main() -> None:
    print("=" * 55)
    print("  Huddle Analysis → Excel Exporter")
    print("=" * 55)
    print("Enter patient IDs separated by spaces or commas.")
    print("Press Enter twice (or Ctrl-D) when done.\n")

    lines: list[str] = []
    try:
        while True:
            line = input("Patient IDs: ").strip()
            if not line and lines:
                break
            if line:
                lines.append(line)
    except EOFError:
        pass

    if not lines:
        print("No patient IDs provided. Exiting.")
        sys.exit(0)

    patient_ids = _parse_ids(" ".join(lines))
    if not patient_ids:
        print("No valid patient IDs found. Exiting.")
        sys.exit(0)

    print(f"\nProcessing {len(patient_ids)} patient(s): {', '.join(patient_ids)}")
    print("-" * 55)

    exporter = HuddleExcelExporter(output_path="data.xlsx")
    exporter.export(patient_ids=patient_ids, json_dir=Path(__file__).parent)


if __name__ == "__main__":
    main()
