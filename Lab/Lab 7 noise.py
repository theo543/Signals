import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy import datasets
from scipy import ndimage
from savefig import savefig

def filter_freq(image: np.ndarray, cutoff: float):
    freq = np.fft.fft2(image)
    xlen, ylen = image.shape
    xcut = int((xlen/2) * cutoff)
    ycut = int((ylen/2) * cutoff)
    freq[xcut:-xcut, ycut:-ycut] *= 0.1
    filtered = np.fft.ifft2(freq).real
    return filtered

def plot(original: np.ndarray, noisy: np.ndarray, ax: Axes):
    noise = np.subtract(original, noisy, dtype=np.float64)
    snr = np.sum(np.square(original, dtype=np.float64)) / np.sum(np.square(noise, dtype=np.float64))
    ax.set_title(f"SNR = {snr:.5f}")
    ax.imshow(noisy, cmap=plt.get_cmap("gray"))
    ax.axis('off')

def main():
    image = datasets.face(gray=True)
    noise_magnitude = 200
    noisy = image + np.random.randint(-noise_magnitude, noise_magnitude + 1, image.shape)
    less_noisy = filter_freq(noisy, 0.1)
    less_noisy_2 = ndimage.gaussian_filter(noisy, 2)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    plot(image, noisy, axs[0])
    plot(image, less_noisy, axs[1])
    plot(image, less_noisy_2, axs[2])
    fig.tight_layout()
    savefig("Lab 7 noise", svg=False)

if __name__ == "__main__":
    main()
