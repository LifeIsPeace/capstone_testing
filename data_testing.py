# This occurs if pandas is imported before torch. If you do then you have to use the os.environ line
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import pandas as pd  # WOW. Import pandas AFTER torch. Or else OpemMP problem happens``
from pathlib import Path
from torch.utils.data import DataLoader
from MaestroDatasetClass import MaestroDataset
from ml_model import PitchNet
from Helper_Files import (masked_bce_loss,
                          validate,
                          collate_fn,
                          Maestro_ROOT,
                          Maestro_ROOT_STR,
                          Maestro_CSV_PATH)
        
        
def slakhdataset():
    SLAKH_PATH: Path = Path(r"D:\capstone_project_data\slakh2100_dataset\synthesized_lakh_dataset\slakh2100_flac_redux")
    splits = ["train", "validation", "test"]
    rows = []
    
    for split in splits:
        split_path: Path = SLAKH_PATH / split
        
        for track_path in split_path.iterdir():
            if not track_path.is_dir():
                continue
    
            # audio file
            audio_path = track_path / "mix.flac"
            
            # MIDI folder
            midi_folder = track_path / "MIDI"
            
            if not midi_folder.exists():
                continue
            
            # loop through the midi files
            for midi_path in midi_folder.glob("*.mid"):
                rows.append({
                    "split": split,
                    "track_id": track_path.name,
                    "audio_path": str(audio_path),
                    "midi_path": str(midi_path)
                })
    
    df = pd.DataFrame(rows)    
    return df

def main():
    
    df = slakhdataset()
    print(df.head())
    print(df["split"].value_counts())
    return
    
    
    df = pd.read_csv(Maestro_CSV_PATH)
    # print(df.head())
    # print(df["split"].value_counts())

    def get_pairs(df, split_type: str):
        subset = df[df["split"] == split_type]
        
        pairs = []
        for _, row in subset.iterrows():
            wav = Maestro_ROOT / row["audio_filename"]
            midi = Maestro_ROOT / row["midi_filename"]
            pairs.append((wav, midi))
            
        return pairs

    # Reduces the amount of data I'm using because good lord is it costly
    # 80/10/10 | 480/60/60
    train_pairs = get_pairs(df, "train")[:10]
    val_pairs = get_pairs(df, "validation")[:1]
    test_pairs = get_pairs(df, "test")[:1]

    # This actually loads the data
    train_ds = MaestroDataset(train_pairs)
    val_ds = MaestroDataset(val_pairs)
    test_ds = MaestroDataset(test_pairs)

    train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True, # Prevent learning order bias
    num_workers=2,
    collate_fn=collate_fn
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False, # No randomness in metrics
    num_workers=2,
    collate_fn=collate_fn
    )

    test_loader = DataLoader(
    test_ds,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
    )

    # Can't use a gpu yet
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PitchNet(n_mels=128, n_pitches=88).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mel, roll, lengths = next(iter(train_loader))
    mel = mel.to(device)
    roll = roll.to(device)
    lengths = lengths.to(device)

    logits = model(mel, lengths)

    print("Mel shape:", mel.shape)
    print("Roll shape:", roll.shape)
    print("Lengths:", lengths.shape)
    print("Logits shape:", logits.shape)

    
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for batch_idx, (mel, roll, lengths) in enumerate(train_loader, 1):
            mel = mel.to(device)
            roll = roll.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(mel, lengths)

            loss = masked_bce_loss(logits, roll, lengths)
            loss.backward()
            optimizer.step()

            # Updates the running average loss
            running_train_loss += loss.item()
            avg_batch_loss = running_train_loss / batch_idx

            # Print running training loss every N batches
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Train Loss: {avg_batch_loss:.4f}"
                )

        # Validation at the end of epoch
        val_loss, val_precision, val_recall, val_f1 = validate(model, val_loader, device)

        # Summary of the epoch
        print(
            f"\nEpoch {epoch+1} Summary | "
            f"Train Loss: {avg_batch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val F1: {val_f1:.4f}\n"
        )

    model.eval()

    mel, roll, lengths = next(iter(test_loader))

    mel = mel.to(device)
    roll = roll.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = model(mel, lengths)
        probs = torch.sigmoid(logits)

    print("Prediction shape:", probs.shape)
    
    
if __name__ == "__main__":
    main()


