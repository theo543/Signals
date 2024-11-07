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

POLY_FILE = Path("Lab 6 polynomials.npz")

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
    if not POLY_FILE.exists():
        poly_benchmark()
    benchmark = np.load(POLY_FILE)
    plt.plot(benchmark['sizes'], benchmark['slow'])
    plt.plot(benchmark['sizes'], benchmark['fast'])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(["Direct Multiplication", "FFT Multiplication"])
    savefig("Lab 6 polynomials")
    plt.close()

if __name__ == "__main__":
    main()
