import librosa
import os
import pathlib
import numpy as np
import soundfile as sf

root_dir = "mswc_microset/"

def get_all_files():
    files = []
    for path in pathlib.Path(root_dir).rglob("*.opus"):
        files.append(path)
    return files

def get_random_file():
    return np.random.choice(get_all_files())

rand_file = get_random_file()
print(rand_file)
X, sr = librosa.load(rand_file)
sf.write("test.wav", X, sr, format="ogg")
