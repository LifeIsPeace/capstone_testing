# This occurs if pandas is imported before torch. If you do then you have to use the os.environ line
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import pandas as pd  # WOW. Import pandas AFTER torch. Or else OpemMP problem happens``
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MaestroDatasetClass import MaestroDataset
from ml_model import PitchNet
from sklearn.model_selection import KFold
from Helper_Files import (masked_bce_loss,
                          validate,
                          collate_fn,
                          train_one_epoch,
                          Maestro_ROOT,
                          Maestro_ROOT_STR,
                          Maestro_CSV_PATH)
        
        
def get_slakh_df():
    SLAKH_PATH = Path(r"D:\capstone_project_data\slakh2100_dataset\synthesized_lakh_dataset\slakh2100_flac_redux")
    splits = ["train", "validation", "test"]

    rows = []

    for split in splits:
        split_path = SLAKH_PATH / split

        for track_path in split_path.iterdir():
            if not track_path.is_dir():
                continue

            audio_path = track_path / "mix.flac"
            midi_folder = track_path / "MIDI"

            if not midi_folder.exists():
                continue

            for midi_path in midi_folder.glob("*.mid"):
                rows.append({
                    "dataset": "slakh",
                    "track_id": track_path.name,
                    "audio_path": str(audio_path),
                    "midi_path": str(midi_path)
                })

    return pd.DataFrame(rows)

def get_maestro_df(csv_path, root_path):
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "dataset": "maestro",
            "track_id": row["audio_filename"],
            "audio_path": str(Path(root_path) / row["audio_filename"]),
            "midi_path": str(Path(root_path) / row["midi_filename"])
        })

    return pd.DataFrame(rows)

def main():
    
    slakh_df = get_slakh_df()
    maestro_df = get_maestro_df(Maestro_CSV_PATH, Maestro_ROOT)

    full_df = pd.concat([slakh_df, maestro_df], ignore_index=True)
    full_df.columns = ["dataset", "track_id", "audio_path", "midi_path"]

    pairs = list(zip(
    full_df["audio_path"],
    full_df["midi_path"],
    full_df["dataset"]
    ))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 2

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []

    all_fold_histories = []  # store curves per fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(pairs)))):
        print(f"\n===== Fold {fold+1} =====")
        
        train_losses = []
        val_losses = []
        val_f1s = []

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]

        train_dataset = MaestroDataset(train_pairs)
        val_dataset = MaestroDataset(val_pairs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn
        )

        # NEW model per fold
        model = PitchNet(n_mels=128, n_pitches=88).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_precision, val_recall, val_f1 = validate(model, val_loader, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            print(f"Fold {fold+1} | Epoch {epoch+1}")
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"F1: {val_f1:.4f}"
            )
        
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title(f"Fold {fold+1} Loss")
        plt.show()

        plt.figure()
        plt.plot(val_f1s, label="Val F1")
        plt.legend()
        plt.title(f"Fold {fold+1} F1")
        plt.show()
        
        fold_results.append(val_f1)
        all_fold_histories.append({
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_f1": val_f1s
        })
        
    print("\n===== Cross-Validation Results =====")
    print(f"Mean F1: {np.mean(fold_results):.4f}")
    print(f"Std F1:  {np.std(fold_results):.4f}")
    
if __name__ == "__main__":
    main()


