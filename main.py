"""Interactive entrypoint for patient huddle analysis."""

from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import DEFAULT_HUDDLE_MODEL, resolve_repo_path
from src.services.huddle_analyzer import HuddleAnalyzer


def _resolve_patient_json_path() -> str:
    candidates = [resolve_repo_path("patients.json"), resolve_repo_path("patient.json")]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def main() -> None:
    patient_id = input("Enter patientId: ").strip()
    if not patient_id:
        print("patientId is required.")
        return
    combined_choice = input("Run combined multi-report lab analysis? [Y/n]: ").strip().lower()
    enable_combined_lab_analysis = combined_choice not in {"n", "no"}

    analyzer = HuddleAnalyzer()
    patients_json = _resolve_patient_json_path()
    output_dir = str(resolve_repo_path("."))

    try:
        analyzer.analyze_patient_huddle(
            patient_id=patient_id,
            patients_json_path=patients_json,
            output_dir=output_dir,
            model=DEFAULT_HUDDLE_MODEL,
            use_web_search=True,
            use_llm_tools=True,
            enable_combined_lab_analysis=enable_combined_lab_analysis,
        )
        print(f"Done. Created file: {patient_id}.json")
    except ValueError as exc:
        print(str(exc))
    except Exception as exc:
        print(f"Failed to run huddle analysis: {exc}")


if __name__ == "__main__":
    main()
