import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.io import wavfile

from TL import *

# signal1 = gf(15, 31, 10)
# plt.show()
# signal2 = gf(15, 310, 100)
# plt.show()
# signal3 = gf(15, 15, 5)
# plt.show()

def dse(file_name, slice_size, start_idx):
    fe, signal = wavfile.read(file_name)
    f, spectre = welch(signal[start_idx:start_idx+slice_size], fe, nperseg=1024)
    spectre = 10 * np.log10(spectre)
    return f, spectre

f, spectre = dse("audacity/CHHHAAAAAAAAA.wav", 2000, 1500)
plt.semilogy(f, spectre)
plt.show()

def zero_padding(signal, pad):
    for i in range(pad):
        signal.insert(0, 0)
        signal.append(0)
    return signal

def decimate(file_name, slice_size, start_idx, n, new_file_name):
    fe, signal = wavfile.read(file_name)
    new_signal = []
    for i in range(0, slice_size, n):
        new_signal.append(signal[start_idx:start_idx+slice_size][i])
    new_fe = int(fe / n)
    wavfile.write(new_file_name, new_fe, np.array(new_signal))
    return new_fe, new_signal

def elevation(file_name, slice_size, start_idx, n, new_file_name):
    fe, signal = wavfile.read(file_name)
    signal = signal[start_idx:start_idx+slice_size]
    for i in range(slice_size, 0, -1):
        for j in range(n):
            signal.insert(i, 0)
    new_fe = int(fe*n)
    wavfile.write(new_file_name, new_fe, np.array(signal))
    return new_fe, signal


def sum_sin(A0, A1, f0, f1, fe, n):
    t = np.arange(0, n/fe, 1/fe)
    y = A0 * np.sin(2 * np.pi * f0 * t) + A1 * np.sin(2 * np.pi * f1 * t)
    # plt.plot(t, y)
    # plt.xlabel("Temps (s)")
    # plt.ylabel("Amplitude")
    # plt.savefig("./plots/q1.1.1.png")
    # plt.show()
    return t, y

def fenetrage_de_hann(t, signal, T):
    new_signal = []
    for i in range(len(t)):
        if t[i] > T:
            new_signal.append(0)
        else:
            new_signal.append(signal[i]*((1/2)-(1/2)*np.cos(2*np.pi*t[i]/T)))
    return new_signal

