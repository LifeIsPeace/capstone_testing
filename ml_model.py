import torch.nn as nn

class PitchNet(nn.Module):
    def __init__(self, n_mels=128, n_pitches=88):
        super().__init__()
        
        # BatchNorm + ReLU
        # The issue is negative values and relu. We want to prevent that because negatives values are 0 for relu soooo
        # we batchnorm it first
        
        # The name conv_stack is arbitrary. It passes this into the forward function
        self.conv_stack = nn.Sequential(
            # 1 in channel (audio), 32 out channels, padding for edge spatial features(Trusting old me's notes)
            # We use 2d to simply capture audio features. Conv3d captures volume of data. 3d might be
            # better but let's keep it simple. Bigger kernel for start. Smaller as the data becomes
            # more dense
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            # Normalizing helps improve training by handling covariance shift
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

        # This is for deeper features
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

    # mel is a tensor
    def forward(self, mel):
        """
        mel: (B, 128, T)
        returns: (B, 88, T)
        B: Batch size
        T: Number of time frames
        """
        x = mel.unsqueeze(1) # (B, 1, 128, T)
        # Conv2D expects input (B, Channels, Height, Width)
        # This reduces frequency but preserves the time dimension
        x = self.conv_stack(x) # (B, 64, 64, T)
        # This further compresses the spectral representation
        x = self.conv_stack2(x)  # (B, 128, 32, T)

        # Unpack the dimensions
        B, C, F, T = x.shape
        # Reorder the dimensions. We want to treat each time frame independently
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        # This flattens all frequency and channel frequencies into one vector
        # Each time frame now has a learned spectral embedding
        # Think of it as (128 channels x 32 frequency bins) per time frame or
        # or 4096 feature vectors per time frame
        x = x.reshape(B, T, C * F)  # (B, T, 128*32)

        # Remember .fc = nn.Linear
        # 88 note logits. Remember logits are not probabilities
        x = self.fc(x)  # (B, T, 88) because paino roll labels are (B, 88, T)
        x = x.permute(0, 2, 1)  # (B, 88, T)

        return x
    