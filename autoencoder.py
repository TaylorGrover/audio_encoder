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
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import time
import zipfile



plt.ion() # Matplotlib interactive mode for viewing plots

np.random.seed(int(time.time())) # Got same audio every single time not sure why. Tryna increase randomness

EPOCHS = 5
CHART_DIR = "training_charts"

def get_all_files(extension):
    filepaths = []
    for path in pathlib.Path(".").rglob("data/*/*.{}".format(extension)):
        filepaths.append(path)
    return filepaths


def convert_audio(filepaths):
    """
    Convert opus to ogg for better support from librosa. 
    Very time-consuming (one-time use), but can be stopped using CTRL-z. 
    Restarting the process starts from where it left off. 
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
    zipped_fname = f"raw_audio_{sr}.zip"
    audio_fname = f"raw_audio_{sr}.npy"
    audio = []
    np.random.shuffle(filepaths)
    start = time.time()
    random_indices = [i for i in range(num_vectors)]
    np.random.shuffle(random_indices)
    '''with ProcessPoolExecutor(max_workers=3) as exe:
        completed = exe.map(get_single_file, filepaths[:num_vectors], repeat(sr), chunksize=40)
        for item in completed:
            if item.shape[0] == sr:
                audio.append(item)'''
    if os.path.isfile(zipped_fname):
        with zipfile.ZipFile(zipped_fname, "r") as zip_ref:
            with zip_ref.open(audio_fname) as unzipped:
                audio = np.load(unzipped)
                print(time.time() - start)
                return audio[random_indices]
    for i, path in enumerate(filepaths):
        if i >= num_vectors:
            break
        X, sample_rate = librosa.load(path, sr=sr)
        if X.shape[0] != sample_rate:
            X = librosa.util.fix_length(X, size=sr)
        audio.append(X)
    print(time.time() - start)
    with zipfile.ZipFile(zipped_fname, "w") as zip_ref:
        np.save(audio_fname.replace(".npy", ""), audio)
        zip_ref.write(audio_fname)
    return np.array(audio, dtype=np.float32)[random_indices]


def show_rand_gram(melspectrograms):
    index = np.random.randint(0, len(melspectrograms))
    spect = melspectrograms[index].reshape(melspectrograms.shape[1:-1])
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(spect, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()


def get_conv_model(input_shape, latent_units = 50):
    Input = tf.keras.Input(shape=input_shape)
    #encoder = Sequential()
    x = layers.Conv2D(32, (5, 5), input_shape=input_shape, activation="relu", padding="same")(Input)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.BatchNormalization(-1)(x)
    x = layers.Conv2D(32, (5, 5), input_shape=input_shape, activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)
    #print(encoder.summary())
    #latent = encoder(Input)
    
    #decoder = Sequential()
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="relu", padding="same")(x)
    #print(decoder.summary())

    model = Model(inputs=Input, outputs=x)
    model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
    return model


def get_model(input_size, latent_units=50):
    """
    This is an attempt to use a model that takes the raw amplitudes
    """
    model = Sequential()
    model.add(layers.Dense(input_size, input_shape=(input_size,), activation="relu"))
    #model.add(layers.BatchNormalization(-1))
    model.add(layers.Dense(1100, activation="tanh"))
    model.add(layers.Dense(latent_units, activation="tanh"))
    model.add(layers.Dense(1100, activation="tanh"))
    model.add(layers.Dense(input_size, activation="linear"))
    model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
    return model


def plot_history(name, history, metric):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history[metric])
    ax.plot(history.history["val_{}".format(metric)])
    ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"], loc="upper left")
    fig.savefig(os.path.join(CHART_DIR, name + ".png"))
    return fig, ax


def conv_data(audio, sr):
    ### Long operation: convert raw audio to melspectrogram
    start = time.time()
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    maximum = np.max(mel)
    print(time.time() - start)

    ### Reshape mel spectrogram so each [row][col] is a singleton array (for keras)
    mel_reshaped = mel.reshape(*mel.shape, 1) / maximum

    X_train, X_test = train_test_split(mel_reshaped, test_size=0.3)
    return X_train, X_test, maximum


def method_convolution(audio, sr=22050):
    X_train, X_test, maximum = conv_data(audio, sr)
    ### Get model
    model = get_conv_model(X_train.shape[1:], latent_units=200)
    model.fit(X_train, X_train, epochs=EPOCHS, validation_split=0.2, batch_size=48, shuffle=True)
    return model, X_train, X_test


def method_dense(audio, sr=22050, latent=100):
    model = get_model(audio.shape[1], latent_units=latent)
    X_train, X_test = train_test_split(audio, test_size=0.3)
    history = model.fit(X_train, X_train, epochs=EPOCHS, validation_split=0.2, batch_size=48, shuffle=True)
    fig, ax = plot_history(f"audio_{sr}", history, "accuracy")
    print(model.evaluate(X_test, X_test))
    return model, X_train, X_test


if __name__ == "__main__":
    sr = 10000 
    filepaths = get_all_files("wav")
    #convert_audio(filepaths)
    audio = get_audio(filepaths, num_vectors=3000, sr=sr)

    #model, X_train, X_test = method_convolution(sr, audio)
    model, X_train, X_test = method_dense(audio, sr, latent=400)
    pred = model.predict(X_test)
    sf.write("orig.wav", audio[0], sr)
    sf.write("pred.wav", pred[0] * 30, sr) # Amplitude scaling necessary

    #stft = librosa.stft(audio)
    #S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    #fig, ax = plt.subplots(figsize=(10, 5))
    '''img = librosa.display.specshow(S_db[10], 
        x_axis="time", 
        y_axis="log", 
        ax=ax
    )'''
    #fig.colorbar(img, ax=ax, format=f'%0.2f')
