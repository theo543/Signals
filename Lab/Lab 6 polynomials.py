from sys import argv, stdout
from time import perf_counter
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def time_fn(fn, *args):
    start = perf_counter()
    result = fn(*args)
    duration = perf_counter() - start
    return duration, result

def direct_poly_mul(p, q):
    assert p.size == q.size
    N = p.size
    pq = np.zeros(N * 2 - 1)
    for i in range(N):
        pq[i:N+i] += p * q[i]
    return pq

def fft_poly_mul(p, q):
    assert p.size == q.size
    p = np.pad(p, (0, p.size))
    q = np.pad(q, (0, q.size))
    return np.fft.ifft(np.fft.fft(p) * np.fft.fft(q))[:-1].real

def poly_benchmark(file: Path, repeats: int):
    sizes = [2 ** i for i in range(0, 16 + 1)]
    slow = np.empty(shape=(len(sizes), repeats))
    fast = np.empty_like(slow)
    for size_idx, size in enumerate(sizes):
        for repeat in range(repeats):
            MAX = 2**31
            p = np.random.randint(-MAX, MAX, size=size, dtype=np.int64)
            q = np.random.randint(-MAX, MAX, size=size, dtype=np.int64)
            t_slow, slow_result = time_fn(direct_poly_mul, p, q)
            t_fast, fast_result = time_fn(fft_poly_mul, p, q)
            assert np.allclose(slow_result, fast_result)
            slow[size_idx][repeat] = t_slow
            fast[size_idx][repeat] = t_fast
            print(f"Repeat {repeat:0>2} of size {p.size: >5} - direct: {t_slow:.5f}, FFT: {t_fast:.5f}", end='\r' if stdout.isatty() else '\n', flush=True)
    np.savez(file, sizes=np.array(sizes), slow=np.array(slow), fast=np.array(fast))
    print(f"\nGenerated {file}", flush=True)

def plot_with_error(sizes, data):
    plt.errorbar(sizes, np.average(data, axis=1), np.std(data, axis=1))
    plt.plot(sizes, np.min(data, axis=1))

def process_benchmark(file: Path, repeats: int, auto_run: bool):
    if not file.exists():
        if auto_run:
            poly_benchmark(file, repeats)
        else:
            print(f"{file} does not exist.")
            return
    benchmark = np.load(file)
    plot_with_error(benchmark['sizes'], benchmark['slow'])
    plot_with_error(benchmark['sizes'], benchmark['fast'])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(["Direct Max Speed", "FFT Max Speed", "Direct Average Speed", "FFT Average Speed"])
    savefig("Lab 6 polynomials")
    plt.close()

def main():
    process_benchmark(Path("Lab 6 polynomials.npz"), 20, True)
    process_benchmark(Path("Lab 6 polynomials (2500).npz"), 2500, "--run-big-benchmark" in argv)

if __name__ == "__main__":
    main()
