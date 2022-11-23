"""
"""

import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import librosa
import os
import pathlib
from queue import Queue
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
    Very time-consuming (one-time use), but can be stopped using CTRL-z. 
    Restarting the process starts from where it left off
    """
    for path in filepaths:
        orig = str(path)
        newpath = orig.replace(".opus", ".ogg")
        if not os.path.isfile(newpath):
            os.system(f"ffmpeg -i {path} -c:a libvorbis {newpath} -hide_banner")


def get_single_file(path, sr=22050):
    X, sr = librosa.load(path, sr=sr)
    return X


def get_audio(filepaths, num_vectors=100, sr=22050):
    """
    Currently really slow: approximately 34 files read per second, but there are
    over 100,000 files
    """
    audio = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=3) as exe:
        completed = exe.map(get_single_file, filepaths, chunksize=40, initargs=(sr,))
        for item in completed:
            if item.shape[0] == sr:
                audio.append(item)
    print(time.time() - start)
    return np.array(audio, dtype=np.float16)

if __name__ == "__main__":
    filepaths = get_all_files("ogg")
    #convert_audio(filepaths)
    audio = get_audio(filepaths, num_vectors=1500)
