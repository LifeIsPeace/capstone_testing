import librosa
import sounddevice as sd

print(librosa.show_versions())

y, sr = librosa.load(librosa.ex('choice'))

sd.play(y, sr)
sd.wait()
