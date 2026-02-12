import librosa
import numpy as np
import pretty_midi as pm
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

# keep them alphabetical. It makes me feel organized 

# n_mels is the number of Mel frequency bands used when converting audio into a Mel spectrogram
# 128 is good for memory while also containing good detail of the audio
# Note the midi has 128 pitches and mel spectrogram has 128 freq bands. There's commonly a mapping between them
class MaestroDataset(Dataset):
    def __init__(
        self,
        pairs, # (wav_path, midi_path)
        sample_rate = 16000, # In hz
        n_mels = 128, # Mel Frequency resolution essentially
        hop_length = 512, # Number of audio samples between consecutive frames
        min_midi = 21, # Lowest MIDI note (A0 lowest key on piano)
        max_midi = 108, # Highest MIDI note (C8 Highest key on piano)
        # This for padding short clips and randomly cropping long clips
        segment_frames = 310, # Fixed number of time frames per training sample
    ):
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.segment_frames = segment_frames
        

        # Frames per second for both Mel and piano roll (mel & labels)
        # Ensures the audio and MIDI are time-aligned
        self.fs = int(round(sample_rate / hop_length))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wav_path, midi_path = self.pairs[idx]

        # Audio is a Raw waveform tensor (np.float32 array)
        audio, sr = sf.read(wav_path)
        audio = torch.tensor(audio).float()
        
        if audio.ndim == 2: 
            audio = audio.mean(dim=1) # stereo to mono

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, sr, self.sample_rate
            )

        audio = audio.numpy().astype(np.float32)

        # Remember Mel spectrogram are log-scaled (human hearing gets worst the higher the freq is)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            fmin=27.5,    # A0
            fmax=4186.0   # C8
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.from_numpy(mel).float()  # (n_mels, T) remember this is the shape

        # This parses the MIDI file into a PrettyMIDI object which has too much stuff to explain in a comment
        midi = pm.PrettyMIDI(midi_path)
        # Piano roll representation
        # Initial shape (128, T) -> (88, T)
        # THis discretizes MIDI time into fs frame per second
        roll = midi.get_piano_roll(fs=self.fs)

        # Restrict to piano range (A0–C8)
        roll = roll[self.min_midi : self.max_midi + 1]

        # Before this, roll is roll[p, t]: p is pitch (MIDI, p=60 is middle c for example) and t is time frame index
        # roll[p, t] is in [0,127] where 0 is no note and any note greater than 0 is its max velocity during that frame 
        roll = (roll > 0).astype(np.float32) # True when any note is active, false otherwise. Produces bool array as float (0.0, 1.0)
        roll = torch.from_numpy(roll) # Now it's a pytorch tensor

        # T is the number of frames in the mel spectrogram
        # mel.shape[0] is the frequency axis (mel bins)
        # mel.shape[1] is the time axis
        T = mel.shape[1]
        
        # Look back and edit maybe
        if T < self.segment_frames:
            # If the pad is too short
            # Essentially adds silence
            pad = self.segment_frames - T # How many extra frames are missing
            mel = torch.nn.functional.pad(mel, (0, pad)) # Pad adds 0 to the end of frame. Test its effects
            roll = torch.nn.functional.pad(roll, (0, pad))
        else:
            # Random crop
            start = torch.randint(0, T - self.segment_frames + 1, (1,)).item()
            mel = mel[:, start:start + self.segment_frames]
            roll = roll[:, start:start + self.segment_frames]

        return mel, roll
    
    # mel: (128, T)
    # roll: (88, T)