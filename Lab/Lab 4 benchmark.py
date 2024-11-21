from time import perf_counter
from pathlib import Path
from sys import argv, stdout
import numpy as np
from matplotlib import pyplot as plt
from savefig import savefig
from plot_with_error import plot_with_error

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
    for benchmark in [(Path("Lab 4 benchmark data.npz"), 20), (Path("Lab 4 benchmark data (10000).npz"), 10000)]:
        process_benchmark(*benchmark)

TEST_SIZES = [4, 128, 256, 512, 1024, 2048, 4096, 8192]

def process_benchmark(benchmark_file: Path, repeats_per_test: int):
    if (not benchmark_file.exists()) or "--run-benchmark" in argv:
        if len(argv) == 3 and argv[1] == "--run-benchmark":
            repeats_per_test = int(argv[2])
        dft_times = np.zeros(shape=(len(TEST_SIZES), repeats_per_test))
        fft_times = dft_times.copy()
        for size_idx, size in enumerate(TEST_SIZES):
            for repeat in range(repeats_per_test):
                end = "\r" if stdout.isatty() else "\n"
                print(f"Running repeat {repeat:02d} of size {size}.", end=end, flush=True)
                signal = np.random.random(size=size)
                dft_result, dft_duration = time_function(dft, signal)
                fft_result, fft_duration = time_function(np.fft.fft, signal)
                assert np.allclose(dft_result, fft_result)
                dft_times[size_idx][repeat] = dft_duration
                fft_times[size_idx][repeat] = fft_duration
        np.savez(benchmark_file, dft=dft_times, fft=fft_times)
        print("\nDone running benchmarks.", flush=True)
    benchmark_data = np.load(benchmark_file)
    plot_with_error(TEST_SIZES, benchmark_data['dft'])
    plot_with_error(TEST_SIZES, benchmark_data['fft'])
    plt.legend(["DFT Fastest", "FFT Fastest", "DFT Average", "FFT Average"])
    plt.yscale("log")
    plt.title(f"DFT vs FFT Benchmark\n{repeats_per_test} repeats per test")
    savefig(f"Lab 4 - 1 ({repeats_per_test} repeats)")
    plt.close()

if __name__ == "__main__":
    main()
