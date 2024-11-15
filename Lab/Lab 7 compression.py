import numpy as np
from matplotlib import pyplot as plt
from scipy import datasets
from savefig import savefig

def remove_freq(freq: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray] | None:
    freq_comp = freq.copy()
    xlen, ylen = freq.shape
    xcut = int((xlen/2)*cutoff)
    ycut = int((ylen/2)*cutoff)
    if xcut == 0 or ycut == 0:
        return None
    freq_comp[xcut:-xcut, ycut:-ycut] = 0
    im_comp = np.fft.ifft2(freq_comp).real
    return freq_comp, im_comp

def compress(image: np.ndarray, minimum_snr: float) -> tuple[float, np.ndarray]:
    PRECISION = 5000
    freq = np.fft.fft2(image)
    l = 1
    r = PRECISION
    while l != r:
        m = (l + r) // 2
        compressed = remove_freq(freq, m / PRECISION)
        if compressed is None:
            l = m + 1
            continue
        _, compressed_imag = compressed
        noise = np.subtract(image, compressed_imag, dtype=np.float64)
        snr = np.sum(np.square(image, dtype=np.float64)) / (np.sum(np.square(noise, dtype=np.float64)) + 10**-10)
        print(f"{m}/{PRECISION} ~= {m/PRECISION:.4f} -> SNR = {snr}")
        if snr >= minimum_snr:
            r = m
        else:
            l = m + 1
    cutoff = l / PRECISION
    compressed = remove_freq(freq, cutoff)
    assert compressed is not None
    compressed_freq, compressed_imag = compressed
    percent_kept = np.count_nonzero(compressed_freq) / np.count_nonzero(image)
    return percent_kept, compressed_imag

def main():
    for name, image in [("ascent", datasets.ascent()), ("face", datasets.face(gray=True))]:
        for snr in [10 ** i for i in [2, 3, 4]]:
            percent_kept, compressed_image = compress(image, snr)
            plt.imshow(compressed_image, cmap=plt.get_cmap('gray'))
            plt.axis('off')
            plt.suptitle(f"Minimum SNR = {snr:g} - kept {percent_kept * 100:.2f}% of frequencies")
            plt.tight_layout()
            savefig(f"Lab 7 compression {name} {snr}", svg=False)
            plt.close()

if __name__ == "__main__":
    main()
