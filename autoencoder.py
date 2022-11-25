"""
"""

import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
import librosa
import librosa.display
import os
import pathlib
from queue import Queue
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import soundfile as sf
import time
import zipfile

plt.ion()

def get_all_files(extension):
    filepaths = []
    for path in pathlib.Path(".").rglob("data/*/0_*.{}".format(extension)):
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
    zipped_fname = "raw_audio.zip"
    audio = []
    np.random.shuffle(filepaths)
    start = time.time()
    '''with ProcessPoolExecutor(max_workers=3) as exe:
        completed = exe.map(get_single_file, filepaths[:num_vectors], repeat(sr), chunksize=40)
        for item in completed:
            if item.shape[0] == sr:
                audio.append(item)'''
    if os.path.isfile(zipped_fname):
        with zipfile.ZipFile(zipped_fname, "r") as zip_ref:
            with zip_ref.open(zipped_fname.replace(".zip", ".npy")) as unzipped:
                audio = np.load(unzipped)
                print(time.time() - start)
                return audio
    for path in filepaths:
        X, sample_rate = librosa.load(path, sr=sr)
        if X.shape[0] != sample_rate:
            X = librosa.util.fix_length(X, size=sr)
        audio.append(X)
    print(time.time() - start)
    return np.array(audio, dtype=np.float16)


def show_rand_gram(melspectrograms):
    index = np.random.randint(0, len(melspectrograms))
    spect = melspectrograms[index].reshape(melspectrograms.shape[1:-1])
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(spect, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()


def get_conv_model(input_shape, latent_units = 50):
    Input = tf.keras.layers.Input(shape=input_shape)
    encoder = Sequential()
    encoder.add(layers.Conv2D(64, (10, 10), input_shape=input_shape, activation="relu"))
    encoder.add(layers.MaxPool2D((2, 2)))
    encoder.add(layers.BatchNormalization(-1))
    encoder.add(layers.Conv2D(34, (5, 5), activation="relu"))
    encoder.add(layers.MaxPool2D((2, 2)))
    encoder.add(layers.BatchNormalization(-1))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(latent_units, activation="relu"))
    
    decoder = Sequential()
    decoder.add(layers.Dense(latent_units, input_shape=(latent_units,), activation="relu"))
    decoder.add(layers.Dense(latent_units * 2, activation="relu"))
    decoder.add(layers.Dense(np.prod(input_shape), activation="relu"))
    decoder.add(layers.Reshape(input_shape))

    latent = encoder(Input)
    output = decoder(latent)
    model = Model(inputs=Input, outputs=output)
    model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
    return model


def get_model(input_size, latent_units=50):
    model = Sequential()
    model.add(layers.Dense(input_size, input_shape=(input_size,), activation="relu"))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dense(500, activation="relu"))
    model.add(layers.Dense(latent_units, activation="sigmoid"))
    model.add(layers.Dense(500, activation="relu"))
    model.add(layers.Dense(input_size, activation="tanh"))
    model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
    return model


def conv_data(audio, sr):
    ### Long operation: convert raw audio to melspectrogram
    start = time.time()
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    print(time.time() - start)

    ### Reshape mel spectrogram so each [row][col] is a singleton array (for keras)
    mel_reshaped = mel.reshape(*mel.shape, 1)
    X_train, X_test = train_test_split(mel_reshaped, test_size=0.3)
    return X_train, X_test


if __name__ == "__main__":
    sr = 22050
    filepaths = get_all_files("wav")
    #convert_audio(filepaths)
    audio = get_audio(filepaths, num_vectors=3000, sr=sr)

    ### Get data
    X_train, X_test = conv_data(audio, sr)

    ### Get model
    model = get_conv_model(X_train.shape[1:], latent_units=200)
    model.fit(X_train, X_train, epochs=5, validation_split=0.2, batch_size=48, shuffle=True)

    #stft = librosa.stft(audio)
    #S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    #fig, ax = plt.subplots(figsize=(10, 5))
    '''img = librosa.display.specshow(S_db[10], 
        x_axis="time", 
        y_axis="log", 
        ax=ax
    )'''
    #fig.colorbar(img, ax=ax, format=f'%0.2f')
