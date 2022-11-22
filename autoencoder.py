"""
"""

import concurrent
from concurrent.futures import ThreadPoolExecutor
import librosa
import os
import pathlib
import numpy as np
import soundfile as sf
import time

def get_all_files(extension):
    filepaths = []
    for path in pathlib.Path(".").rglob("*.{}".format(extension)):
        filepaths.append(path)
    return filepaths


def convert_audio(filepaths):
    """
    Convert opus to ogg for better support from librosa. 
    Very time-consuming (one-time use)
    """
    for path in filepaths:
        orig = str(path)
        newpath = orig.replace(".opus", ".ogg")
        if not os.path.isfile(newpath):
            os.system(f"ffmpeg -i {path} -c:a libvorbis {newpath}")


def get_single_file(path, sr):
    X, sr = librosa.load(path, sr=sr)
    return X


def get_audio(filepaths, num_vectors=100):
    audio = []
    start = time.time()
    sr = 3675
    with ThreadPoolExecutor() as executor:
        results = []
        for i, path in enumerate(filepaths):
            if i >= num_vectors:
                break
            results.append(executor.submit(get_single_file, path, sr))
        #results = [executor.submit(get_single_file, path, sr) for path in filepaths]
        for result in concurrent.futures.as_completed(results):
            audio.append(result.result())
    #X, sr = librosa.load(path, sr=3675)
    #audio.append(X)
    print(time.time() - start)
    return np.array(audio), result

if __name__ == "__main__":
    filepaths = get_all_files("ogg")
    #convert_audio(filepaths)
    audio, result = get_audio(filepaths, num_vectors=500)
