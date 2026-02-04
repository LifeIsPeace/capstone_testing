import pandas as pd
from pathlib import Path

    
ROOT_STR: str = r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0"
ROOT: Path = Path(r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0")
CSV_PATH: Path = ROOT / "maestro-v3.0.0.csv"

df = pd.read_csv(CSV_PATH)

print(df.head())
print(df["split"].value_counts())
