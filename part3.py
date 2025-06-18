import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy.io import wavfile

from TL import *

# signal1 = gf(15, 31, 10)
# plt.show()
# signal2 = gf(15, 310, 100)
# plt.show()
# signal3 = gf(15, 15, 5)
# plt.show()


def zero_padding(signal, pad):
    """Rajoute pad zéros au début et à la fin du signal."""
    signal = list(signal)
    for i in range(pad):
        signal.insert(0, 0)
        signal.append(0)
    return np.array(signal)


def dse(file_name, slice_size, start_idx, pad = None):
    """Renvoie la densité spectrale d'énergie du signal."""
    fe, signal = wavfile.read(file_name)
    if pad is not None:
        signal = zero_padding(signal, pad)
    f, spectre = welch(signal[start_idx:start_idx+slice_size], fe, nperseg=1024)
    spectre = 10 * np.log10(spectre)
    return f, spectre


f, spectre = dse("audacity/CHHHAAAAAAAAA.wav", 10000, 21000, 10000)
plt.semilogy(f, spectre)
plt.show()



def decimate(file_name, slice_size, start_idx, n, new_file_name):
    """Décime le signal avec un facteur n."""
    fe, signal = wavfile.read(file_name)
    new_signal = []
    for i in range(0, slice_size, n):
        new_signal.append(signal[start_idx:start_idx+slice_size][i])
    new_fe = int(fe / n)
    wavfile.write(new_file_name, new_fe, np.array(new_signal))
    return new_fe, new_signal

def elevation(file_name, slice_size, start_idx, n, new_file_name):
    "Elevation de signal de facteur n."
    fe, signal = wavfile.read(file_name)
    signal = list(signal[start_idx:start_idx+slice_size])
    for i in range(slice_size, 0, -1):
        for j in range(n):
            signal.insert(i, 0)
    new_fe = int(fe*n)
    wavfile.write(new_file_name, new_fe, np.array(signal))
    return new_fe, np.array(signal)


down = decimate("audacity/CHHHAAAAAAAAA.wav", 10000, 21000, 2, "down_chat.wav")
up = elevation("audacity/CHHHAAAAAAAAA.wav", 10000, 21000, 2, "up_chat.wav")

Ndown = len(down[1])
yfdown = fft(down[1])
xfdown = fftfreq(Ndown, 1/down[0])[:Ndown//2]
Nup = len(up[1])
yfup = fft(up[1])
xfup = fftfreq(Nup, 1/up[0])[Nup//2]

plt.plot(xfup, 2/Nup * np.abs(yfup[0:Nup//2]))
plt.plot(xfdown, 2/Ndown * np.abs(yfdown[0:Ndown//2]))
plt.grid()
plt.show()



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

