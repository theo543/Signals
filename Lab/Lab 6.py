from time import perf_counter
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def timed_fn(fn):
    def fn_with_time(*args):
        start = perf_counter()
        result = fn(*args)
        duration = perf_counter() - start
        return duration, result
    return fn_with_time

@timed_fn
def direct_poly_mul(p, q):
    assert p.size == q.size
    N = p.size
    pq = np.zeros(N * 2 - 1)
    for i in range(N):
        pq[i:N+i] += p * q[i]
    return pq

@timed_fn
def fft_poly_mul(p, q):
    assert p.size == q.size
    p = np.pad(p, (0, p.size - 1))
    q = np.pad(q, (0, q.size - 1))
    return np.fft.ifft(np.fft.fft(p) * np.fft.fft(q)).real

def rectangle(n):
    return np.abs(np.arange(n) - n / 2) < n / 4

def hanning(n):
    return (1 - np.cos(np.linspace(0, 2 * np.pi, num=n))) / 2

POLY_FILE = Path("Lab 6 polynomial benchmark.npz")

def poly_benchmark():
    slow = []
    fast = []
    sizes = [2 ** i for i in range(1, 16 + 1)]
    for size in sizes:
        MAX = 2**31
        p = np.random.randint(-MAX, MAX, size=size, dtype=np.int64)
        q = np.random.randint(-MAX, MAX, size=size, dtype=np.int64)
        t1, pq1 = direct_poly_mul(p, q)
        t2, pq2 = fft_poly_mul(p, q)
        assert np.allclose(pq1, pq2)
        slow.append(t1)
        fast.append(t2)
    np.savez(POLY_FILE, sizes=np.array(sizes), slow=np.array(slow), fast=np.array(fast))

def main():
    x = np.random.random(size=100)
    fig, axs = plt.subplots(nrows=4, ncols=1)
    for w in range(4):
        axs[w].plot(x)
        x = np.convolve(x, x)
    fig.tight_layout()
    savefig("Lab 6 - 1")
    plt.close()

    if not POLY_FILE.exists():
        poly_benchmark()
    benchmark = np.load(POLY_FILE)
    plt.plot(benchmark['sizes'], benchmark['slow'])
    plt.plot(benchmark['sizes'], benchmark['fast'])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(["Direct Multiplication", "FFT Multiplication"])
    savefig("Lab 6 - 2")
    plt.close()

    x = np.linspace(0, 1, num=1_000_000)
    sin = np.sin(2 * np.pi * x * 100)
    plt.plot(sin * rectangle(x.size))
    plt.plot(sin * hanning(x.size))
    plt.legend(["Rectangle", "Hanning"])
    savefig("Lab 6 - 3")
    plt.close()

    data = np.genfromtxt("Lab 5 Data.csv", delimiter=",", skip_header=1, usecols=(2,), max_rows=72)
    fig, axs = plt.subplots(nrows=5, ncols=1)
    axs[0].plot(data)
    for i, w in enumerate([5, 10, 15, 20]):
        smooth = np.convolve(data, np.ones(w), 'valid')
        axs[i + 1].plot(smooth)
    fig.tight_layout()
    savefig("Lab 6 - 4 b")

if __name__ == "__main__":
    main()
