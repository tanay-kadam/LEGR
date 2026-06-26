from pathlib import Path

import pandas as pd


def read_datafile(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file, auto-detected by extension."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
