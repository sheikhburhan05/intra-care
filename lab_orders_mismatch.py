"""Check lab orders rows where enterprise ID does not match patient ID."""

from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poc.config import resolve_repo_path


def read_lab_orders_and_log_mismatch(csv_path: str) -> pd.DataFrame:
    """Read Lab orders CSV and log rows where enterpriseid != patientid."""
    dataframe = pd.read_csv(csv_path)
    mismatch = dataframe[dataframe["enterpriseid"] != dataframe["patientid"]]
    print(f"Lab orders total rows: {len(dataframe)}")
    print(f"Rows where enterpriseid != patientid: {len(mismatch)}")
    if len(mismatch) > 0:
        print("\nMismatched rows:")
        print(mismatch.to_string())
    else:
        print("\nNo mismatches found.")
    return mismatch


if __name__ == "__main__":
    path = resolve_repo_path(Path("data") / "Lab orders.csv")
    read_lab_orders_and_log_mismatch(str(path))
