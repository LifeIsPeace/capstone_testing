import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from MaestroDatasetClass import MaestroDataset
from ml_model import PitchNet


def masked_bce_loss(logits, targets, lengths):
    """
    logits:  (B, 88, T): Raw model outputs
    targets: (B, 88, T): 0 or 1 is it on or not
    lengths: (B,): Real sequence lengths before padding
    """

    # B: Batch size
    # T: Maximum time dimension
    B, _, T = logits.shape

    # .arange creates time indices
    # True (for <) if real timestep, false if padded
    mask = torch.arange(T, device=lengths.device).expand(B, T) < lengths.unsqueeze(1)
    # Add pitch dimension
    mask = mask.unsqueeze(1)  # (B, 1, T)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    # Loss at padded positions become 0
    # Real timesteps should remain unchanged
    loss = loss * mask
    # Total loss over valid elements / number of valid (non padded) time positions
    return loss.sum() / mask.sum()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (mel, roll, lengths) in enumerate(loader):
        mel = mel.to(device)
        roll = roll.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        logits = model(mel, lengths)

        loss = masked_bce_loss(logits, roll, lengths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return running_loss / len(loader)


# Save my storage good lord
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    
    running_loss = 0.0
    
    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0

    for mel, roll, lengths in loader:
        mel = mel.to(device)
        roll = roll.to(device)
        lengths = lengths.to(device)

        logits = model(mel, lengths)

        # Loss
        loss = masked_bce_loss(logits, roll, lengths)
        running_loss += loss.item()

        # Metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        B, _, T = logits.shape

        mask = torch.arange(T, device=lengths.device).expand(B, T) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)

        preds = preds * mask
        roll = roll * mask

        TP = (preds * roll).sum()
        FP = (preds * (1 - roll)).sum()
        FN = ((1 - preds) * roll).sum()

        total_TP += TP.item()
        total_FP += FP.item()
        total_FN += FN.item()

    # dataset level metrics here
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_loss = running_loss / len(loader)

    print(f"\nValidation Results")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return avg_loss, precision, recall, f1
        

    """_summary_ 
    My samples had different time lengths
    Ex: mel1: (n_mels, 500); mels2: (nmels, 620)
    
    I couldn't stack them because of the different dimensions
    SOOOOOO
    - This finds the longest sequance in the batch, 
    - Pads everything to that length
    - Returns stacked tensors + original lengths
    """
def collate_fn(batch):
    mels, rolls = zip(*batch)

    lengths = torch.tensor([m.shape[1] for m in mels])

    max_T = max(lengths)

    padded_mels = []
    padded_rolls = []

    # This just computes the required padding
    for mel, roll in zip(mels, rolls):
        pad = max_T - mel.shape[1]
        mel = F.pad(mel, (0, pad))
        roll = F.pad(roll, (0, pad))

        padded_mels.append(mel)
        padded_rolls.append(roll)

    # This stacks it into batch tensors
    return (
        torch.stack(padded_mels),
        torch.stack(padded_rolls),
        lengths
    )

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
    # 80/10/10
    train_pairs = get_pairs(df, "train")[:480]
    val_pairs = get_pairs(df, "validation")[:60]
    test_pairs = get_pairs(df, "test")[:60]

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


