import numpy as np
import matplotlib.pyplot as plt


def gf(f0, fe, n):
    t = np.arange(0, n, 1/fe)
    y = np.sin(2 * np.pi * f0 * t)
    plt.plot(t, y)
    plt.show()
    return (t, y)


