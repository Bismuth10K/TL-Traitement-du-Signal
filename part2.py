from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, fft
from TL import *

sinus = gf(20, 350, 40)[1]
trig = triangle(20, 3)[1]
bruit = noise(200, 3)[1]
plt.show()
plt.close()

# autocorrelate(sinus)
# autocorrelate(trig)
# autocorrelate(bruit)

aaa = wavfile.read("./audacity/aaa.wav")[1]

aaa = aaa / 32767
print(max(aaa))

# autocorrelate(aaa[500:950])
# autocorrelate(aaa[3400:3950])
# autocorrelate(aaa[4230:4810])
# autocorrelate(aaa[7500:8280])

# autocorrelate(aaa)

chat = wavfile.read("./audacity/CHHHAAAAAAAAA.wav")[1]
chat = chat / 32767

# autocorrelate(chat[200:900])
# autocorrelate(chat[10150:10830])

# autocorrelate(chat[21200:22010])
# autocorrelate(chat[34500:35320])

signal = chat[21200:22010]
freq = np.arange(len(signal))/len(signal)*16000
fft1 = fft(signal)

tfft = 2**20
fft2 = 10*np.log10(np.abs(fft(signal, tfft))/len(signal)*2)
freq2 = np.arange(tfft)/tfft*16000

plt.semilogy(freq2, np.absolute(fft2))
plt.show()

