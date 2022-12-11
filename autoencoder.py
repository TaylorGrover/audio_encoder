"""
"""

import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import glob
from itertools import repeat
from keras_visualizer import visualizer
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
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

EPOCHS = 10
MODEL_DIR = "models"
CHART_DIR = "training_charts"


class WeightViewer(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        weight = {}
        for layer in self.model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)

def get_all_files(extension, only_zero=True):
    filepaths = []
    if only_zero:
        pathglob = f"data/*/0*.{extension}"
    else:
        pathglob = f"data/*/*.{extension}"
    return glob.glob(pathglob)


def get_checkpoint(name):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, name),
        monitor="val_loss",
        mode="min",
        save_best_only=True
    )


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
                print(len(audio))
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


def get_model(input_size, latent_units=50):
    """
    This model reads in standard-scaled audio amplitudes with columns removed
    """
    model = Sequential()
    model.add(layers.Dense(input_size, input_shape=(input_size,), activation="relu"))
    model.add(layers.Dense(latent_units + 250, activation="tanh"))
    model.add(layers.Dropout(.2))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dense(latent_units + 50, activation="tanh"))
    model.add(layers.Dropout(.2))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dense(latent_units, activation="tanh"))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dense(latent_units + 50, activation="tanh"))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(input_size, activation="linear"))
    schedule = optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-1,
        end_learning_rate=1e-4,
        decay_steps=10000, 
        power=1.0,
    )
    opt = optimizers.Adam(learning_rate=schedule)
    model.compile(optimizer="adadelta", loss="MSE")
    return model


def get_conv_model(input_shape, latent_units = 50):
    Input = tf.keras.Input(shape=input_shape)
    #encoder = Sequential()
    x = layers.Conv2D(64, (3, 3), input_shape=input_shape, activation="leaky_relu", padding="same")(Input)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.BatchNormalization(-1)(x)
    #print(encoder.summary())
    #latent = encoder(Input)
    
    #decoder = Sequential()
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="leaky_relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="linear", padding="same")(x)
    #print(decoder.summary())

    model = Model(inputs=Input, outputs=x)
    print(model.summary())
    model.compile(optimizer="adam", loss="MSE")
    return model


def plot_history(name, history, metric):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history[metric])
    ax.plot(history.history["val_{}".format(metric)])
    ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"], loc="upper left")
    fig.savefig(os.path.join(CHART_DIR, name + "_" + metric + ".png"))
    return fig, ax


def conv_data(audio, sr, mels=64):
    ### Long operation: convert raw audio to melspectrogram
    start = time.time()
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=mels)
    db = librosa.amplitude_to_db(mel, ref=np.max)
    print(time.time() - start)

    ### Reshape mel spectrogram so each [row][col] is a singleton array (for keras)
    db = db.reshape(*db.shape, 1) # This may not be a good idea

    X_train, X_test = train_test_split(db, test_size=0.3)
    return X_train, X_test


def method_convolution(audio, sr=22050, latent=200):
    X_train, X_test = conv_data(audio, sr)
    ### Get model
    model = get_conv_model(X_train.shape[1:], latent_units=200)
    model_name = f"mel_{sr}_{latent}_{model.loss}"
    checkpoint_callback = get_checkpoint(model_name)
    weight_callback = WeightViewer(model)
    weight_callback.on_epoch_end(-1)
    history = model.fit(X_train, X_train, epochs=EPOCHS, validation_split=0.2, batch_size=48, shuffle=True, callbacks=[checkpoint_callback, weight_callback])
    fig, ax = plot_history(model_name, history, "loss")
    return model, X_train, X_test


def method_dense(audio, sr=22050, latent=100):
    model = get_model(audio.shape[1], latent_units=latent)
    X_train, X_test = train_test_split(audio, test_size=0.3)
    model_name = f"audio_{sr}_{latent}_{model.loss}"
    checkpoint_callback = get_checkpoint(model_name)
    history = model.fit(
        X_train, 
        X_train, 
        epochs=EPOCHS, 
        validation_split=0.1, 
        batch_size=48,
        shuffle=True,
        callbacks=[checkpoint_callback],
    )
    fig, ax = plot_history(model_name, history, "loss")
    fig, ax = plot_history(model_name, history, "accuracy")
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
    pred = model.predict(X_test)
    sf.write("orig.wav", audio[0] * 200, sr)
    sf.write("pred.wav", pred[0] * 200, sr) # Amplitude scaling necessary
    print(model.evaluate(X_test, X_test))
    return model, X_train, X_test


def plot_rand_predict(pred, actual, ax1, ax2):
    index = np.random.randint(0, len(pred))
    librosa.display.specshow(pred[index], x_axis="time", y_axis="log", ax=ax1)
    librosa.display.specshow(actual[index], x_axis="time", y_axis="log", ax=ax2)


def preprocess_audio(audio):
    """
    Use sklearn StandardScaler to give columns mean 0 and stddev 1
    """
    scaler = StandardScaler().fit(audio)
    res = scaler.transform(audio)
    return res, scaler


if __name__ == "__main__":
    sr = 8000 
    filepaths = get_all_files("wav")
    #convert_audio(filepaths)
    audio = get_audio(filepaths, num_vectors=3000, sr=sr) # You can use more than 3000 vectors for 
    #scaled, scaler = preprocess_audio(audio)
    #print(type(audio))

    model, X_train, X_test = method_convolution(audio, sr)
    #model, X_train, X_test = method_dense(audio, sr, latent=1100)

    #stft = librosa.stft(audio)
    #S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    #fig, ax = plt.subplots(figsize=(10, 5))
    '''img = librosa.display.specshow(S_db[10], 
        x_axis="time", 
        y_axis="log", 
        ax=ax
    )'''
    #fig.colorbar(img, ax=ax, format=f'%0.2f')
