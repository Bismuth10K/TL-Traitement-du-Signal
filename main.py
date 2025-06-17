from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from TL import *


if __name__ == "__main__":
    samplerate_bj, data_bj = wavfile.read('./audacity/bonjour.wav')
    samplerate_ar, data_ar = wavfile.read('./audacity/aurevoir.wav')
    # print(data_bj)
    # print(samplerate_bj)
    # print(samplerate_ar)

    N = 16
    data_ar = data_ar / (2**N/2)
    data_bj = data_bj / (2**N/2)

    fig, axs = plt.subplots(2)
    t_bj = np.arange(len(data_bj)) / samplerate_bj
    t_ar = np.arange(len(data_ar)) / samplerate_ar
    axs[0].plot(t_bj, data_bj)
    axs[1].plot(t_ar, data_ar)
    plt.savefig("plots/q1.2.2.png")
    plt.show()

    print(min(data_bj), max(data_bj))
    print(min(data_ar), max(data_ar))

    #! Q 1.2.3
    new_t_bj3, new_y_bj3 = quantify(t_bj, data_bj, 3, "q1.2.3-bj")
    wavfile.write('audios/bj3.wav', samplerate_bj, np.array(new_y_bj3))
    new_t_bj8, new_y_bj8 = quantify(t_bj, data_bj, 8, "q1.2.3-bj")
    wavfile.write('audios/bj8.wav', samplerate_bj, np.array(new_y_bj8))

    new_t_ar3, new_y_ar3 = quantify(t_ar, data_ar, 3, "q1.2.3-ar")
    wavfile.write("audios/ar3.wav", samplerate_ar, np.array(new_y_ar3))
    new_t_ar8, new_y_ar8 = quantify(t_ar, data_ar, 8, "q1.2.3-ar")
    wavfile.write("audios/ar8.wav", samplerate_ar, np.array(new_y_ar8))

    print(noise_energy(new_t_bj3, 3))
    print(noise_energy(new_t_bj3, 8))
    print(noise_energy(new_t_ar3, 3))
    print(noise_energy(new_t_ar3, 8))
