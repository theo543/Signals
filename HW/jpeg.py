from dataclasses import dataclass
from typing import Self, BinaryIO
from itertools import combinations
from heapq import heapify, heappop, heappush
import numpy as np
import numba
from scipy.datasets import face
from scipy.fft import dctn, idctn

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
    assert len(img.shape) in [2, 3]
    rows, cols = img.shape[:2]
    row_blk = (rows + 7) // 8
    col_blk = (cols + 7) // 8
    if len(img.shape) == 2:
        result = np.empty((row_blk, col_blk, 8, 8), dtype=img.dtype)
    else:
        result = np.empty((row_blk, col_blk, 8, 8, 3), dtype=img.dtype)
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
    assert len(img.shape) in [4, 5]
    row_blk, col_blk = img.shape[:2]
    if len(img.shape) == 4:
        result = np.empty((row_blk * 8, col_blk * 8), dtype=img.dtype)
    else:
        result = np.empty((row_blk * 8, col_blk * 8, 3), dtype=img.dtype)
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
    S = 5000 / quality if quality < 50 else 200 - 2 * quality
    Q = np.floor((S * Q + 50) / 100).astype(np.uint16)
    Q[Q == 0] = 1
    return Q

def quantize(img, Q):
    img /= Q
    np.round(img, out=img)
    return img.astype(np.int16)

INVERSE_ZIGZAG_IDX = np.array([
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
])

ZIGZAG_IDX = np.argsort(INVERSE_ZIGZAG_IDX)

def zigzag(img: np.ndarray):
    if len(img.shape) == 4:
        return img.reshape(img.shape[0], img.shape[1], -1)[:, :, ZIGZAG_IDX].copy()
    return img.reshape(img.shape[0], img.shape[1], -1, 3)[:, :, ZIGZAG_IDX, :].copy()

@numba.jit(cache=True)
def dc_delta_encode(img: np.ndarray):
    blocks = img.reshape(-1, 64)
    prev = blocks[0, 0]
    for block in blocks[1:]:
        dc = block[0]
        block[0] = dc - prev
        prev = dc

@numba.jit(cache=True)
def dc_delta_decode(img: np.ndarray):
    blocks = img.reshape(-1, 64)
    prev = blocks[0, 0]
    for block in blocks[1:]:
        dc = block[0]
        block[0] = dc + prev
        prev = block[0]

@numba.jit(cache=True)
def encoding_size(value: np.ndarray):
    shape = value.shape
    value = value.ravel()
    sizes = np.empty_like(value, dtype=np.int16)
    for i, val in enumerate(value):
        if val == 0:
            sizes[i] = 0
        else:
            sizes[i] = int(np.log2(np.abs(val))) + 1
    return sizes.reshape(shape)

ZRL = 0xF0
EOB = 0x00

@numba.jit(cache=True)
def count_codewords(zz_img: np.ndarray, dc_count: np.ndarray, ac_count: np.ndarray):
    assert len(zz_img.shape) == 3
    assert zz_img.shape[-1] == 64
    assert dc_count.shape == (12,)
    assert ac_count.shape == (256,)

    blocks = zz_img.reshape(-1, 64)
    sizes = encoding_size(blocks)

    dc_count += np.bincount(sizes[:, 0], minlength=12)

    ac_count[EOB] += sizes.shape[0] - np.count_nonzero(sizes[:, -1])

    for ac_block in sizes[:, 1:]:
        zeros = 0
        for size in np.trim_zeros(ac_block, trim="b"):
            if size == 0:
                zeros += 1
                if zeros == 16:
                    ac_count[ZRL] += 1
                    zeros = 0
                continue
            ac_count[(zeros << 4) | size] += 1
            zeros = 0

def build_huffman_table(symbol_count: np.ndarray, subtree_integrity_debug_checks=False) -> tuple[np.ndarray, np.ndarray]:
    # All-ones code is forbidden in JPEG.
    # A dummy node with count 0 is used to ensure no real node is assigned all-ones.
    # Any real node must take at least one left edge in the Huffman tree.
    symbol_count = np.append(symbol_count, 0)
    dummy_idx = symbol_count.size - 1

    next_in_subtree = np.full_like(symbol_count, fill_value=-1)

    @dataclass(slots=True, frozen=True)
    class Subtree:
        first: int
        weight: int
        has_dummy: bool

        def __lt__(self, other: Self) -> bool:
            if self.weight == other.weight:
                return self.has_dummy
            return self.weight < other.weight

        def __post_init__(self):
            if not subtree_integrity_debug_checks:
                return
            weight = 0
            node = self.first
            has_dummy = False
            while node != -1:
                has_dummy = has_dummy or node == dummy_idx
                weight += symbol_count[node]
                node = next_in_subtree[node]
            assert weight == self.weight
            assert has_dummy == self.has_dummy

    # Codeword length is equal to depth in the Huffman tree.
    codeword_length = np.zeros_like(symbol_count)

    # Min-heap of subtrees.
    subtrees = [Subtree(i, count, i == dummy_idx) for i, count in enumerate(symbol_count) if i == dummy_idx or count > 0]
    heapify(subtrees)

    while len(subtrees) > 1:
        a = heappop(subtrees)
        b = heappop(subtrees)

        node = a.first

        while True:
            codeword_length[node] += 1
            next_node = next_in_subtree[node]
            if next_node == -1:
                break
            node = next_node

        next_in_subtree[node] = b.first
        node = b.first

        while node != -1:
            codeword_length[node] += 1
            node = next_in_subtree[node]

        heappush(subtrees, Subtree(a.first, a.weight + b.weight, a.has_dummy or b.has_dummy))

    #  Dummy node should have maximum length.
    assert codeword_length[-1] == np.max(codeword_length[:-1])

    huffman_count = np.bincount(codeword_length, minlength=33)
    huffman_values = np.argsort(codeword_length)[huffman_count[0]:]
    huffman_count[0] = 0

    # There shouldn't be codewords longer than 32 bits.
    assert huffman_count.size == 33

    assert huffman_count.max() <= 256
    huffman_count = huffman_count.astype(np.uint8)

    return huffman_count, huffman_values

def adjust_huffman_counts(huffman_count: np.ndarray, huffman_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    assert huffman_count.shape == (33,)
    assert huffman_count.sum() == huffman_values.size
    assert huffman_values.size > 0

    # Codewords are limited to 16 bits using the Adjust_BITS algorithm described in the standard (section K.3).
    # In Huffman codes there is always a pair of symbols with a common prefix in the longest category.
    # Symbols are removed two at a time from the longest category.
    # The prefix for the pair is used for one of the symbols.
    # To make space for the second symbol, a codeword from a shorter category becomes a prefix for two longer codewords.

    i = 32
    while i > 16:
        if huffman_count[i] > 0:
            j = i - 2
            while huffman_count[j] == 0:
                j -= 1
            huffman_count[i] -= 2
            huffman_count[i - 1] += 1
            huffman_count[j + 1] += 2
            huffman_count[j] -= 1
        else:
            i = i - 1

    # Remove dummy value.
    while huffman_count[i] == 0:
        i -= 1
    huffman_count[i] -= 1
    huffman_values = huffman_values[:-1]
    assert huffman_values.max() < 256
    huffman_values = huffman_values.astype(np.uint8)

    assert np.all(huffman_count[17:] == 0)
    huffman_count = huffman_count[:17]
    assert huffman_count.sum() == huffman_values.size

    return huffman_count, huffman_values

def build_adjusted_huffman_table(symbol_count: np.ndarray, subtree_integrity_debug_checks=False):
    return adjust_huffman_counts(*build_huffman_table(symbol_count, subtree_integrity_debug_checks))

@numba.jit(cache=True)
def expand_huffman_table(huffman_count: np.ndarray, huffman_values: np.ndarray):
    # expanded_table[value] = [codeword_length, codeword]
    expanded_table = np.zeros(shape=(256, 2), dtype=np.uint16)

    value_idx = 0
    next_codeword = 0
    for codeword_length, count in enumerate(huffman_count):
        for _ in range(count):
            expanded_table[huffman_values[value_idx]] = [codeword_length, next_codeword]
            value_idx += 1
            next_codeword += 1
        next_codeword <<= 1

    return expanded_table

def test_huffman_table(N=20):
    random = np.random.default_rng()

    def test_build_table(count):
        table = build_adjusted_huffman_table(count, subtree_integrity_debug_checks=True)
        expanded = expand_huffman_table(*table)
        binary_str = [f"{val:0{length}b}" for [length, val] in expanded if length != 0]
        for bin1, bin2 in combinations(binary_str, 2):
            assert not bin1.startswith(bin2)
            assert not bin2.startswith(bin1)

    test_build_table(np.array([1]))

    for _ in range(N):
        test_build_table(random.integers(0, 1_000, 256))

        test = random.integers(0, 1_000, 256)
        test[random.integers(0, 256)] = 1_000_000
        test_build_table(test)

        test = random.integers(0, 1_000, 256)
        test *= random.integers(0, 2, 256)
        test_build_table(test)

def build_huffman_tables(zz_img: np.ndarray):
    # JPEG has separate tables for Y component and for Cb+Cr components.
    assert len(zz_img.shape) == 3 or zz_img.shape[3] == 2

    dc_count = np.zeros(12, dtype=np.int64)
    ac_count = np.zeros(256, dtype=np.int64)

    if len(zz_img.shape) == 3:
        count_codewords(zz_img, dc_count, ac_count)
    else:
        count_codewords(zz_img[:, :, :, 0], dc_count, ac_count)
        count_codewords(zz_img[:, :, :, 1], dc_count, ac_count)

    return build_adjusted_huffman_table(dc_count), build_adjusted_huffman_table(ac_count)

@numba.experimental.jitclass
class ByteBuffer:
    _next_byte: numba.uint64
    _bytebuffer: numba.uint8[:]
    _bitbuffer: numba.uint64
    _bitcapacity: numba.uint8

    def __init__(self):
        self._next_byte = 0
        self._bytebuffer = np.empty(16, dtype=np.uint8)
        self._bitbuffer = 0
        self._bitcapacity = 64

    def _resize(self, new_size):
        new_buffer = np.empty(new_size, dtype=np.uint8)
        new_buffer[:self._bytebuffer.size] = self._bytebuffer
        self._bytebuffer = new_buffer

    def _add_byte(self, byte):
        self._bytebuffer[self._next_byte] = byte
        self._next_byte += 1
        if byte == 0xFF:
            self._bytebuffer[self._next_byte] = 0
            self._next_byte += 1

    def _flush(self):
        if self._next_byte + 16 > self._bytebuffer.size:
            self._resize(self._bytebuffer.size * 2)
        while self._bitcapacity <= 56:
            byte = self._bitbuffer >> 56
            self._add_byte(byte)
            self._bitbuffer <<= 8
            self._bitcapacity += 8

    def final_flush(self) -> np.ndarray:
        self._flush()
        if self._bitcapacity != 64:
            last_byte = self._bitbuffer >> 56
            if self._next_byte + 2 >= self._bytebuffer.size:
                self._resize(self._bytebuffer.size + 2)
            last_byte |= 255 >> (64 - self._bitcapacity)
            self._add_byte(last_byte)
            self._bitcapacity = 64

        return self._bytebuffer[:self._next_byte]

    def add_bits(self, length, bits):
        if length > self._bitcapacity:
            self._flush()
        self._bitbuffer |= bits << (self._bitcapacity - length)
        self._bitcapacity -= length

@numba.jit(cache=True)
def huffman_encode(zz_img: np.ndarray, dc_table: tuple[np.ndarray, np.ndarray], ac_table: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    bytebuffer = ByteBuffer()

    expanded_dc_table = expand_huffman_table(*dc_table)
    expanded_ac_table = expand_huffman_table(*ac_table)

    def add_dc_codeword(value):
        length, codeword = expanded_dc_table[value]
        bytebuffer.add_bits(length, codeword)

    def add_ac_codeword(value):
        length, codeword = expanded_ac_table[value]
        bytebuffer.add_bits(length, codeword)

    flat_zz_img = zz_img.reshape(-1, 64)

    for block in flat_zz_img:
        eob = block[-1] == 0
        block = np.trim_zeros(block, trim='b')
        if block.size == 0:
            add_dc_codeword(0)
            add_ac_codeword(EOB)
            continue
        sizes = encoding_size(block)
        encodings = block + (2 ** sizes - 1) * (block < 0)
        add_dc_codeword(sizes[0])
        bytebuffer.add_bits(sizes[0], encodings[0])
        zeros = 0
        for i in range(1, block.size):
            if sizes[i] == 0:
                zeros += 1
                if zeros == 16:
                    add_ac_codeword(ZRL)
                    zeros = 0
            else:
                assert zeros < 16
                assert sizes[i] < 16
                add_ac_codeword((zeros << 4) | sizes[i])
                bytebuffer.add_bits(sizes[i], encodings[i])
                zeros = 0
        if eob:
            add_ac_codeword(EOB)

    return bytebuffer.final_flush()

def lossy_compress(blocks: np.ndarray, q: np.ndarray):
    dct = dctn(blocks, s=(8, 8), axes=(2, 3), norm="ortho")
    dct /= q
    assert -(2 ** 15) <= dct.min() <= dct.max() <= 2 ** 15
    return np.round(dct, out=dct)

def lossy_uncompress(blocks: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = idctn(blocks * q, s=(8, 8), axes=(2, 3), norm="ortho")
    return np.round(result, out=result)

def lossy_compress_color(img: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray):
    img = to_blocks(img)
    ycbcr = (img @ RGB_TO_YCBCR).round().astype(np.int16)
    ycbcr[:, :, :, :, 0] -= 128
    ycbcr[:, :, :, :, 0] = lossy_compress(ycbcr[:, :, :, :, 0], q_luma)
    ycbcr[:, :, :, :, 1] = lossy_compress(ycbcr[:, :, :, :, 1], q_chroma)
    ycbcr[:, :, :, :, 2] = lossy_compress(ycbcr[:, :, :, :, 2], q_chroma)
    return ycbcr

def lossy_uncompress_color(ycbcr: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray, original_shape):
    ycbcr[:, :, :, :, 0] = lossy_uncompress(ycbcr[:, :, :, :, 0], q_luma)
    ycbcr[:, :, :, :, 0] += 128
    ycbcr[:, :, :, :, 1] = lossy_uncompress(ycbcr[:, :, :, :, 1], q_chroma)
    ycbcr[:, :, :, :, 2] = lossy_uncompress(ycbcr[:, :, :, :, 2], q_chroma)
    img = (ycbcr @ YCBCR_TO_RGB).round()
    img = np.round(img, out=img)
    img = np.clip(img, 0, 255, out=img)
    return from_blocks(img.astype(np.uint8), *original_shape)

def lossy_compress_grayscale(img: np.ndarray, q: np.ndarray):
    img = to_blocks(img.astype(np.int16))
    img -= 128
    return lossy_compress(img, q).astype(np.int16)

def lossy_uncompress_grayscale(img: np.ndarray, q: np.ndarray, original_shape):
    img = lossy_uncompress(img, q)
    img += 128
    return from_blocks(np.clip(img, 0, 255, out=img).astype(np.uint8), *original_shape)

def mse(orig, noisy):
    orig = orig.astype(np.float64)
    noisy = noisy.astype(np.float64)
    return np.mean(np.square(orig - noisy))

def snr(orig, noisy):
    orig = orig.astype(np.float64)
    noisy = noisy.astype(np.float64)
    return np.sum(np.square(orig)) / np.sum(np.square(orig - noisy))

def title(orig, noisy):
    return f"MSE = {mse(orig, noisy):.2f}\nSNR = {snr(orig, noisy):.2f}"

def write_huffman_table(file: BinaryIO, table: tuple[np.ndarray, np.ndarray], is_ac: bool, table_id: int):
    huffman_lengths, huffman_values = table
    assert huffman_lengths.shape == (17,)
    assert huffman_lengths[0] == 0
    assert huffman_values.shape == (huffman_lengths.sum(),)
    assert huffman_values.itemsize == 1
    assert huffman_lengths.itemsize == 1

    file.write(
        b"\xFF\xC4" + # DHT
        int(2 + 1 + 16 + huffman_lengths.sum()).to_bytes(2, 'big') +
        int((is_ac << 4) + table_id).to_bytes(1)
    )
    huffman_lengths[1:].tofile(file, sep="")
    huffman_values.tofile(file, sep="")

def write_quantization_matrix(file: BinaryIO, q: np.ndarray, matrix_id: int):
    assert q.shape == (8, 8)
    assert q.itemsize == 2

    extended = 1
    if q.max() < 256:
        extended = 0
        q = q.astype(np.uint8)

    file.write(
        b"\xFF\xDB" + # DQT
        int(2 + 1 + 64 * q.itemsize).to_bytes(2, 'big') +
        ((extended << 4) + matrix_id).to_bytes(1)
    )
    file.flush()
    q.reshape(64).tofile(file, sep="")

@dataclass
class SOFComponentParameters:
    component_id: int
    horizontal_sampling: int
    vertical_sampling: int
    quantization_table: int

def write_start_of_frame(file: BinaryIO, height: int, width: int, bits_per_sample: int, components: list[SOFComponentParameters]):
    file.write(
        b"\xFF\xC0" + # SOF0
        int(2 + 1 + 2 + 2 + 1 + 3 * len(components)).to_bytes(2, 'big') +
        bits_per_sample.to_bytes(1) +
        height.to_bytes(2, 'big') +
        width.to_bytes(2, 'big') +
        len(components).to_bytes(1)
    )
    for component in components:
        file.write(
            component.component_id.to_bytes(1) +
            ((component.horizontal_sampling << 4) + component.vertical_sampling).to_bytes(1) +
            component.quantization_table.to_bytes(1)
        )

@dataclass
class SOSComponentParameters:
    component_id: int
    dc_table_id: int
    ac_table_id: int

def write_start_of_scan(file: BinaryIO, components: list[SOSComponentParameters]):
    file.write(
        b"\xFF\xDA" + # SOS
        int(2 + 1 + 2 * len(components) + 3).to_bytes(2, 'big') +
        len(components).to_bytes(1)
    )
    for component in components:
        file.write(
            component.component_id.to_bytes(1) +
            ((component.dc_table_id << 4) + component.ac_table_id).to_bytes(1)
        )
    file.write(b"\x00\x3F\x00") # Start of spectral selection = 0, end of spectral selection = 63, successive approximation (set to 0 in sequential JPEG)

def write_app0(file: BinaryIO):
    file.write(
        b"\xFF\xE0" + # APP0
        (2 + 5 + 2 + 1 + 2 + 2 + 2).to_bytes(2, 'big') +
        b"JFIF\0" +
        b"\x01\x02" +
        b"\0" +
        b"\0\0" +
        b"\0\0" +
        b"\0" +
        b"\0"
    )

def write_grayscale_jpeg(file: BinaryIO, height: int, width: int, q_luma: np.ndarray, luma_dc_table: tuple[np.ndarray, np.ndarray], luma_ac_table: tuple[np.ndarray, np.ndarray], entropy_coded_data: np.ndarray):
    file.write(b"\xFF\xD8") # SOI
    write_app0(file)
    write_quantization_matrix(file, q_luma, 0)
    write_start_of_frame(file, height, width, 8, [SOFComponentParameters(1, 1, 1, 0)])
    write_huffman_table(file, luma_dc_table, False, 0)
    write_huffman_table(file, luma_ac_table, True, 0)
    write_start_of_scan(file, [SOSComponentParameters(1, 0, 0)])
    entropy_coded_data.tofile(file, sep="")
    file.write(b"\xFF\xD9") # EOI

def main():
    test_blocks_reversible()
    test_huffman_table()

    img = face(gray=True)
    q_luma = Q_quality(Q_LUMA, 100)
    img_comp = lossy_compress_grayscale(img, q_luma)
    dc_delta_encode(img_comp)
    zz_img = zigzag(img_comp)
    tables = build_huffman_tables(zz_img)
    entropy_coded_data = huffman_encode(zz_img, *tables)
    with open("face.jpg", "wb") as file:
        write_grayscale_jpeg(file, img.shape[0], img.shape[1], q_luma, tables[0], tables[1], entropy_coded_data)

if __name__ == "__main__":
    main()
