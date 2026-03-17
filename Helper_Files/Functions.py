import torch
import torch.nn.functional as F

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
    
