import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def rectangle(n):
    return np.abs(np.arange(n) - n / 2) < n / 4

def hanning(n):
    return (1 - np.cos(np.linspace(0, 2 * np.pi, num=n))) / 2

def main():
    x = np.linspace(0, 1, num=1_000_000)
    sin = np.sin(2 * np.pi * x * 100)
    plt.plot(sin * rectangle(x.size))
    plt.plot(sin * hanning(x.size))
    plt.legend(["Rectangle", "Hanning"])
    savefig("Lab 6 windows")
    plt.close()

if __name__ == "__main__":
    main()
