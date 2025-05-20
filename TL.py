import numpy as np
import matplotlib.pyplot as plt


def gf(f0, fe, n):
    t = np.arange(0, n, 1/fe)
    y = np.sin(2 * np.pi * f0 * t)
    plt.plot(t, y)
    plt.show()
    return t, y


def energy(t, y):
    return np.trapezoid(y**2, t)

sig = gf(0.1,10,20)
# print(energy(sig[0], sig[1]))


def mean_power(t, y, f0, fe):
    stop = int(fe/f0 + 1)
    return np.trapezoid(y[:stop]**2, t[:stop])

# print(mean_power(sig[0], sig[1], 0.1, 10))


def quantify(t, y, n):
    q = 1/2**(n-1)
    tab = np.arange(0,2**n, 1)
    tab2 = (tab - np.mean(tab)) * q
    new_y = []
    print(tab2)
    for pt in y:
        diff = np.abs(pt - tab2)
        new_y.append(tab2[np.argmin(diff)])
    plt.plot(t, new_y)
    plt.show()
    return t, new_y


# quantify(sig[0], sig[1], 2)

def noise_energy(t, n):
    return ((n**2)/12) * (t[-1]-t[0])


def rsb(t, y, f0, fe, n):
    return 10 * np.log10(mean_power(t, y, f0, fe)/noise_energy(t, n))


def autocorrelate(y, tau):
    # on suppose le signal réel
    y_tau = np.zeros(len(y))
    y_tau[:tau] = y[-tau:]
    return sum(y * y_tau)


def residu(y):
    # note numpy n'utilise pas la fft, on ne sait donc pas d'où vient l'écart entre les 2.
    ref = np.correlate(y,y, "full")
    cor = np.zeros(len(y))
    for i in range(1, len(y)-1):
        cor[i] = autocorrelate(y, i)
    tab1 = cor[1:-1]
    tab2 = ref[:len(tab1)]
    plt.plot(tab2 - tab1)
    plt.title("Différence entre np.correlate et TL.autocorrelate")
    plt.ylabel("écart")
    plt.xlabel("tau")
    plt.show()


residu(sig[1])