import torch.nn as nn

class PitchNet(nn.Module):
    def __init__(self, n_mels=128, n_pitches=88):
        super().__init__()
        
        # BatchNorm + ReLU
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # frequency pooling only (it halves the frequency dimension) meaning it does not touch time
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Frequency dimension after pooling:
        # 128 -> 64 -> 32
        self.fc = nn.Linear(128 * 32, n_pitches)

    def forward(self, mel):
        """
        mel: (B, 128, T)
        returns: (B, 88, T)
        """
        x = mel.unsqueeze(1) # (B, 1, 128, T)
        x = self.conv_stack(x)    # (B, 64, 64, T)
        x = self.conv_stack2(x)   # (B, 128, 32, T)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, C * F)  # (B, T, 128*32)

        x = self.fc(x)   # (B, T, 88)
        x = x.permute(0, 2, 1)  # (B, 88, T)

        return x