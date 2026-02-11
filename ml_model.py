import torch.nn as nn

class PitchNet(nn.Module):
    def __init__(self, n_mels=128, n_pitches=88):
        super().__init__()
        
        # BatchNorm + ReLU
        # THe issue is negative values and relu. We want to prevent that because negatives values are 0 for relu soooo
        # we batchnorm it first
        
        # Squential obviously for con2d then batchnorm2d then relu
        self.conv_stack = nn.Sequential(
            # 1 in channel (audio), 32 out channels, padding for edge spatial features(Trusting old me's notes)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # Essentially gets the mean (approx 0) and standard dev (aprox one) across a batch
            nn.BatchNorm2d(32),
            # Gotta deal with disappearing gradient problem so Relu seems good
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # frequency pooling only (it halves the frequency dimension) meaning it does not touch time
            # downsamples the spatial dimensions (retains important feature information)
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # This should be deeper features
        # Brush up on what everything means. I'm going off of AI class last year
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
        x = self.conv_stack(x)  # (B, 64, 64, T)
        x = self.conv_stack2(x)   # (B, 128, 32, T)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, C * F)  # (B, T, 128*32)

        x = self.fc(x)   # (B, T, 88)
        x = x.permute(0, 2, 1)  # (B, 88, T)

        return x