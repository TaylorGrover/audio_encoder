import librosa
import os
import pathlib
import numpy as np

root_dir = "mswc_microset/"

def get_all_files():
    files = []
    for path in pathlib.Path(root_dir).rglob("*.opus"):
        files.append(path)
    return files

files = get_all_files()
