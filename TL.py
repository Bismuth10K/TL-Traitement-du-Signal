import numpy as np
import matplotlib.pyplot as plt


def gf(f0, fe, n):
    t = np.arange(0, n/fe, 1/fe)
    y = np.sin(2 * np.pi * f0 * t)
    plt.plot(t, y)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    # plt.savefig("./plots/q1.1.1.png")
    # plt.show()
    return t, y


def energy(t, y):
    return np.trapezoid(y**2, t)


def mean_power(t, y, f0, fe):
    stop = int(fe/f0 + 1)
    return np.trapezoid(y[:stop]**2, t[:stop])



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
    plt.xlabel("Temps (s)")
    plt.ylabel(f"Signal quantifié sur {n} bits")
    # Rajouter -2 à la fin du titre pour un n = 3
    # plt.savefig("./plots/q1.1.3.png")
    plt.show()
    return t, new_y



def noise_energy(t, n):
    # viens du cours
    return (((1/2**(n-1))**2)/12) * (t[-1]-t[0])


def rsb(t, y, f0, fe, n):
    return 10 * np.log10(mean_power(t, y, f0, fe)/noise_energy(t, n))


def autocorrelate(y, tau):
    # on suppose le signal réel
    y_tau = np.zeros(len(y))
    y_tau[:tau] = y[-tau:]
    return sum(y * y_tau)


def residu(y):
    # note numpy n'utilise pas la fft, on ne sait donc pas d'où vient l'écart entre les 2.
    # TODO : ajouter comparaison avec scipy.correlate
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


def triangle(f0, nb_periode):
    signal = []
    omega = 2 * np.pi * f0
    fe = 4 * f0
    time = np.arange(0, nb_periode + 1/fe, 1/fe)
    for t in time:
        somme = 0
        for i in range(10):
            somme += ((-1)**i) * np.sin((2*i+1)*omega*t) / ((2*i+1)**2)
        signal.append((8/(np.pi ** 2)) * somme)
    return time, signal

# trig = triangle(1, 5)
# plt.plot(trig[0], trig[1])
# plt.show()


def noise(fe, nb_periode):
    time = np.arange(0, nb_periode + 1/fe, 1/fe)
    return time, np.random.normal(0, 0.4, len(time))

# bruit = noise(100, 5)
# plt.plot(bruit[0], bruit[1])
# plt.show()


# pour 3.1

sig1 = gf(10, 21, 200)
sig2 = gf(10, 40, 200)
sig3 = gf(10, 10, 200)

plt.show()