from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from TL import *


if __name__ == "__main__":
    samplerate_bj, data_bj = wavfile.read('./audacity/bonjour.wav')
    samplerate_ar, data_ar = wavfile.read('./audacity/aurevoir.wav')
    # print(data_bj)
    print(samplerate_bj)
    print(samplerate_ar)

    fig, axs = plt.subplots(2)
    t_bj = np.arange(len(data_bj)) / samplerate_bj
    t_ar = np.arange(len(data_ar)) / samplerate_ar
    axs[0].plot(t_bj, data_bj)
    axs[1].plot(t_ar, data_ar)
    plt.show()

    print(min(data_bj))
    print(max(data_bj))

    scaled_data_bj = (((data_bj - min(data_bj)) / (max(data_bj) - min(data_bj))) * 2) - 1
    print(min(scaled_data_bj))
    print(max(scaled_data_bj))

    #! Q 1.2.3
    new_t_bj, new_y_bj = quantify(t_bj, scaled_data_bj, 4)
