
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import soundfile as sf

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

audio_files = glob('zero.wav')
y, sr = librosa.load(audio_files[0])


S = librosa.feature.melspectrogram(y=y,
                                   sr=sr//2,
                                   n_mels=128 * 2,)

file = open("op.txt", 'w+')

print(str(S))
file.write(str(S))
Sh = np.array(S, dtype=np.float16)

Sh2 = np.array(Sh, dtype=np.float32)

ToAud = librosa.feature.inverse.mel_to_stft(Sh2)

y = librosa.griffinlim(ToAud)

y_tritone = librosa.effects.pitch_shift(y, sr=sr, n_steps=9)

sf.write('stereo_file.wav', y, sr, subtype='PCM_24')

sf.write('stereo_fileUP.wav', y_tritone, sr, subtype='PCM_24')

S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(10, 5))
 #Plot the mel spectogram
img = librosa.display.specshow(S_db_mel,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Mel Spectogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()
