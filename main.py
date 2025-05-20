from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from TL import *

samplerate_bj, data_bj = wavfile.read('./audacity/bonjour.wav')
samplerate_ar, data_ar = wavfile.read('./audacity/aurevoir.wav')
print(data_bj)
print(samplerate_bj)
print(samplerate_ar)

fig, axs = plt.subplots(2)
t_bj = np.arange(len(data_bj)) / samplerate_bj
t_ar = np.arange(len(data_ar)) / samplerate_ar
axs[0].plot(t_bj, data_bj)
axs[1].plot(t_ar, data_ar)
plt.show()


#! Q 1.2.3
new_t_bj, new_y_bj = quantify(t_bj, data_bj, 4)
