import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from MaestroDatasetClass import MaestroDataset
from ml_model import PitchNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for mel, roll in loader:
        mel = mel.to(device)     # (B, 128, T)
        roll = roll.to(device)   # (B, 88, T)

        optimizer.zero_grad()

        logits = model(mel)      # (B, 88, T)
        loss = criterion(logits, roll)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for mel, roll in loader:
        mel = mel.to(device)
        roll = roll.to(device)

        logits = model(mel)
        loss = criterion(logits, roll)

        running_loss += loss.item()

    return running_loss / len(loader)

def main():
    ROOT_STR: str = r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0"
    ROOT: Path = Path(r"D:\capstone_project_data\maestro_dataset\maestro-v3.0.0\maestro-v3.0.0")
    CSV_PATH: Path = ROOT / "maestro-v3.0.0.csv"

    df = pd.read_csv(CSV_PATH)

    # print(df.head())
    # print(df["split"].value_counts())

    def get_pairs(df, split_type: str):
        subset = df[df["split"] == split_type]
        
        pairs = []
        for _, row in subset.iterrows():
            wav = ROOT / row["audio_filename"]
            midi = ROOT / row["midi_filename"]
            pairs.append((wav, midi))
            
        return pairs

    # Reduces the amount of data I'm using because good lord is it costly
    train_pairs = get_pairs(df, "train")[:100]
    val_pairs = get_pairs(df, "validation")[:100]
    test_pairs = get_pairs(df, "test")[:20]

    # This actually loads the data

    train_ds = MaestroDataset(train_pairs)
    val_ds = MaestroDataset(val_pairs)
    test_ds = MaestroDataset(test_pairs)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False
    )

    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PitchNet(n_mels=128, n_pitches=88).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mel, roll = next(iter(train_loader))
    mel = mel.to(device)
    roll = roll.to(device)

    logits = model(mel)
    print("Mel shape:   ", mel.shape)
    print("Roll shape:  ", roll.shape)
    print("Logits shape:", logits.shape)

    
    num_epochs = 2

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
    
    
if __name__ == "__main__":
    main()


