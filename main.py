from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

samplerate_bj, data_bj = wavfile.read('./audacity/bonjour.wav')
samplerate_ar, data_ar = wavfile.read('./audacity/aurevoir.wav')

print(samplerate_bj)
print(samplerate_ar)

fig, axs = plt.subplots(2)
axs[0].plot(np.arange(len(data_bj)) / samplerate_bj, data_bj)
axs[1].plot(np.arange(len(data_ar)) / samplerate_ar, data_ar)
plt.show()
