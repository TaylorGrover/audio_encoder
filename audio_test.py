"""
Read a random .opus file and output a wav file of the audio to verify the 
correctness of librosa
"""
import glob
import librosa
import os
import pathlib
import numpy as np
import soundfile as sf

root_dir = "mswc_microset/"

def get_all_files():
    files = glob.glob("data/*/0*.wav")
    return files

def get_random_file():
    return np.random.choice(get_all_files())

rand_file = get_random_file()
print(rand_file)
X, sr = librosa.load(rand_file)
sf.write("test.wav", X, sr, format="wav")
mel = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=64)
db = librosa.amplitude_to_db(mel, ref=np.max)
amp = librosa.db_to_amplitude(db)
out = librosa.feature.inverse.mel_to_audio(amp, sr=sr)
sf.write("noisy.wav", out, sr, format="wav")
