# Path module
Pure Path Useless lol. Unless you're dealing with multiple os's in some way.
</b>Concrete Paths are the way to go

# Maestro dataset in drive
```
- MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1
- MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1

- MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1
- MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1
```

- In the drive, every associated midi and wav file have the same name. All that needs to change
is the file extention.
- In the future, iterate through the folders with training data (2004,2006, etc) by reading if the file begins with a digit
Make sure to store if you have already iterated through it.

# The hard stuff
- Sample rate in this context means "how many audio samples per second". So sample_rate = 16000 means 16000 per second
- The hop length is meant for the mel spectrogram. You slice audio into overlapping frames. Each frame starts 512 samples after the previous one.
- Frames per second makes: one mel frame = one piano-roll frame

# Caching
Make sure to implement caching for speedups