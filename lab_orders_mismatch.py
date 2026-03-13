import pandas as pd
import os

def read_lab_orders_and_log_mismatch(csv_path: str) -> pd.DataFrame:
    """Read Lab orders CSV and log rows where enterpriseid != patientid."""
    df = pd.read_csv(csv_path)
    mismatch = df[df['enterpriseid'] != df['patientid']]
    print(f"Lab orders total rows: {len(df)}")
    print(f"Rows where enterpriseid != patientid: {len(mismatch)}")
    if len(mismatch) > 0:
        print("\nMismatched rows:")
        print(mismatch.to_string())
    else:
        print("\nNo mismatches found.")
    return mismatch

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "Lab orders.csv")
    read_lab_orders_and_log_mismatch(path)
