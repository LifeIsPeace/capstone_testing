import pandas as pd
from pathlib import Path

    
ROOT_STR: str = r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0"
ROOT: Path = Path(r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0")
CSV_PATH: Path = ROOT / "maestro-v3.0.0.csv"

df = pd.read_csv(CSV_PATH)

print(df.head())
print(df["split"].value_counts())

def get_pairs(df, split_type: str):
    subset = df[df["split"] == split_type]
    
    pairs = []
    for _, row in subset.iterrows():
        wav = ROOT / row["audio_filename"]
        midi = ROOT / row["midi_filename"]
        pairs.append((wav, midi))
        
    return pairs

train_pairs = get_pairs(df, "train")
val_pairs = get_pairs(df, "validation")
test_pairs = get_pairs(df, "test")