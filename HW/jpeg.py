import numpy as np
from scipy.datasets import face
from scipy.fft import dctn, idctn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
#from imageio.v3 import imread

RGB_TO_YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
)

YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR)

Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

Q_CHROMA = np.array(
    [
        [9, 9, 12, 24, 50, 50, 50, 50],
        [9, 11, 13, 33, 50, 50, 50, 50],
        [12, 13, 28, 50, 50, 50, 50, 50],
        [24, 33, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
    ]
)


def to_blocks(img):
    assert len(img.shape) == 2
    rows, cols = img.shape
    row_blk = (rows + 7) // 8
    col_blk = (cols + 7) // 8
    result = np.empty((row_blk, col_blk, 8, 8), dtype=img.dtype)
    for row in range(rows // 8):
        for col in range(cols // 8):
            result[row, col] = img[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
        if cols % 8 != 0:
            result[row, -1, :, : cols % 8] = img[row * 8 : (row + 1) * 8, -(cols % 8) :]
            result[row, -1, :, cols % 8 :] = img[row * 8 : (row + 1) * 8, -1:]
    if rows % 8 != 0:
        for col in range(cols // 8):
            result[-1, col, : rows % 8, :] = img[-(rows % 8) :, col * 8 : (col + 1) * 8]
            result[-1, col, rows % 8 :, :] = img[-1:, col * 8 : (col + 1) * 8]
        if cols % 8 != 0:
            result[-1, -1, : rows % 8, : cols % 8] = img[-(rows % 8) :, -(cols % 8) :]
            result[-1, -1, rows % 8 :, : cols % 8] = img[-1:, -(cols % 8) :]
            result[-1, -1, : rows % 8, cols % 8 :] = img[-(rows % 8) :, -1:]
            result[-1, -1, rows % 8 :, cols % 8 :] = img[-1, -1]
    return result


def from_blocks(img, row, col):
    assert len(img.shape) == 4
    assert all(dim == 8 for dim in img.shape[2:4])
    row_blk, col_blk = img.shape[:2]
    result = np.empty((row_blk * 8, col_blk * 8), dtype=img.dtype)
    for i in range(row_blk):
        for j in range(col_blk):
            result[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = img[i][j]
    return result[:row, :col]


def test_blocks_reversible(N=20):
    random = np.random.default_rng()
    for _ in range(N):
        rows, cols = random.integers(1, 1000, 2)
        img = random.integers(0, 255, (rows, cols), dtype=np.uint8)
        assert np.array_equal(img, from_blocks(to_blocks(img), rows, cols))


def Q_quality(Q, quality):
    Q = np.floor(((200 - 2 * quality) * Q + 50) // 100).astype(np.uint8)
    Q[Q == 0] = 1
    return Q


def roundtrip(orig_img, Q):
    img = to_blocks(orig_img)
    nonzero = np.count_nonzero(img)
    img = dctn(img, s=(8, 8), axes=(2, 3), norm="ortho")
    img /= Q
    img = np.round(img)
    set_to_0 = (nonzero - np.count_nonzero(img)) / img.size
    img = img * Q
    img = idctn(img, s=(8, 8), axes=(2, 3), norm="ortho")
    img = from_blocks(img, *orig_img.shape)
    return set_to_0, img

def roundtrip_grayscale(orig_img, quality):
    assert orig_img.dtype == np.uint8
    shifted = orig_img.astype(np.int16) - 128
    q_luma = Q_quality(Q_LUMA, quality)
    set_to_0, img = roundtrip(shifted, q_luma)
    unshifted = np.clip(img + 128, 0, 255).astype(np.uint8)
    return set_to_0, unshifted

def roundtrip_color(orig_img, quality):
    ycbcr = (orig_img @ RGB_TO_YCBCR).astype(np.int16)
    q_luma = Q_quality(Q_LUMA, quality)
    q_chroma = Q_quality(Q_CHROMA, quality)
    set_to_0 = np.empty(3)
    set_to_0[0], ycbcr[:, :, 0] = roundtrip(ycbcr[:, :, 0], q_luma)
    set_to_0[1], ycbcr[:, :, 1] = roundtrip(ycbcr[:, :, 1], q_chroma)
    set_to_0[2], ycbcr[:, :, 2] = roundtrip(ycbcr[:, :, 2], q_chroma)
    img = np.clip(ycbcr @ YCBCR_TO_RGB, 0, 255).astype(np.uint8)
    return np.mean(set_to_0), img

def mse(orig, noisy):
    orig = orig.astype(np.float64)
    noisy = noisy.astype(np.float64)
    return np.mean(np.square(orig - noisy))

def snr(orig, noisy):
    orig = orig.astype(np.float64)
    noisy = noisy.astype(np.float64)
    return np.sum(np.square(orig)) / np.sum(np.square(orig - noisy))

def title(orig, noisy, set_to_0):
    return f"MSE = {mse(orig, noisy):.2f}\nSNR = {snr(orig, noisy):.2f}\nZeroed Frequencies = {set_to_0 * 100:.0f}%"

def main():
    test_blocks_reversible()
    quality = 75
    orig_color = face(gray=False)
    orig_gray = face(gray=True)
    set_to_0_color, color = roundtrip_color(orig_color, quality)
    set_to_0_gray, gray = roundtrip_grayscale(orig_gray, quality)
    axs : list[Axes]
    _, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_title(title(orig_gray, gray, set_to_0_gray))
    axs[0].imshow(gray, cmap=plt.get_cmap("gray"))
    axs[1].set_title(title(orig_color, color, set_to_0_color))
    axs[1].imshow(color)
    plt.savefig("roundtrip results.png")


if __name__ == "__main__":
    main()
