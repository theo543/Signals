from time import perf_counter
from pathlib import Path
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig

def time_function(fn, argument):
    start_time = perf_counter()
    result = fn(argument)
    return (result, perf_counter() - start_time)

def dft(signal):
    N = signal.size
    row = np.arange(0, N)
    col = row[:, np.newaxis]
    w = np.e ** (-2 * np.pi * 1j / N)
    F = w ** (row * col)
    return signal @ F

def main():
    benchmark_data = Path("Lab 4 benchmark data.npz")
    test_sizes = [4, 128, 256, 512, 1024, 2048, 4096, 8192]
    if (not benchmark_data.exists()) or "--run-benchmark" in argv:
        repeats_per_test = 20
        dft_times = np.zeros(shape=(len(test_sizes), repeats_per_test))
        fft_times = dft_times.copy()
        for size_idx, size in enumerate(test_sizes):
            for repeat in range(repeats_per_test):
                print(f"Running repeat {repeat:02d} of size {size}.", end="\r")
                signal = np.random.random(size=size)
                dft_result, dft_duration = time_function(dft, signal)
                fft_result, fft_duration = time_function(np.fft.fft, signal)
                assert np.allclose(dft_result, fft_result)
                dft_times[size_idx][repeat] = dft_duration
                fft_times[size_idx][repeat] = fft_duration
        np.savez(benchmark_data, dft=dft_times, fft=fft_times)
        print("\nDone running benchmarks.")
    benchmark_data = np.load(benchmark_data)
    dft_times = benchmark_data['dft']
    fft_times = benchmark_data['fft']
    def plot_with_error(times):
        mean = np.mean(times, axis=1)
        std = np.std(times, axis=1)
        plt.errorbar(test_sizes, mean, yerr=std)
    plot_with_error(dft_times)
    plot_with_error(fft_times)
    plt.legend(["DFT", "FFT"])
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"DFT vs FFT Benchmark\n{dft_times.shape[1]} repeats per test")
    savefig("Lab 4 - 1")
    plt.close()

if __name__ == "__main__":
    main()
