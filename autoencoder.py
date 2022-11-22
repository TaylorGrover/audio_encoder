"""
"""

import librosa
import os
import pathlib
import numpy as np
import soundfile as sf
import time

def get_all_files():
    filepaths = []
    for path in pathlib.Path(".").rglob("*.opus"):
        filepaths.append(path)
    return filepaths

def get_audio(filepaths):
    audio = []
    start = time.time()
    for path in filepaths:
        X, sr = librosa.load(path)
        audio.append(X)
    print(time.time() - start)
    return np.array(audio)

if __name__ == "__main__":
    filepaths = get_all_files()
    audio = get_audio(filepaths)
