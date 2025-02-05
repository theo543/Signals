from dataclasses import dataclass
from typing import Any, Self, BinaryIO, cast
from itertools import combinations, product
from heapq import heapify, heappop, heappush
from sys import byteorder
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from argparse import ArgumentParser
import numpy as np
import numba
from scipy.datasets import face
from scipy.fft import dctn, idctn
from skvideo.io import FFmpegReader
from imageio.v3 import imread

RGB_TO_YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
).T

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
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
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
    assert len(img.shape) == 4
    assert img.shape[2] == 8 and img.shape[3] == 8
    return img.reshape(-1, 64)[:, ZIGZAG_IDX].copy()

@numba.jit(cache=True)
def dc_delta_encode(img: np.ndarray):
    blocks = img.reshape(-1, 64)
    prev = blocks[0, 0]
    for block in blocks[1:]:
        dc = block[0]
        block[0] = dc - prev
        prev = dc

@numba.jit(cache=True)
def dc_delta_encode_color(img: np.ndarray):
    blocks = img.reshape(-1, 64, 3)
    prevs = [blocks[0, 0, 0], blocks[0, 0, 1], blocks[0, 0, 2]]
    for block in blocks[1:]:
        for i in range(3):
            dc = block[0, i]
            block[0, i] = dc - prevs[i]
            prevs[i] = dc

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
    assert len(zz_img.shape) == 2
    assert zz_img.shape[-1] == 64
    assert dc_count.shape == (12,)
    assert ac_count.shape == (256,)

    sizes = encoding_size(zz_img)

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

Table = tuple[np.ndarray, np.ndarray]

def build_huffman_table(symbol_count: np.ndarray, subtree_integrity_debug_checks=False) -> Table:
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

def adjust_huffman_counts(huffman_count: np.ndarray, huffman_values: np.ndarray) -> Table:

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

def build_huffman_tables(zz_img: np.ndarray, zz_img_2: np.ndarray | None = None):

    dc_count = np.zeros(12, dtype=np.int64)
    ac_count = np.zeros(256, dtype=np.int64)

    count_codewords(zz_img, dc_count, ac_count)
    if zz_img_2 is not None:
        count_codewords(zz_img_2, dc_count, ac_count)

    return build_adjusted_huffman_table(dc_count), build_adjusted_huffman_table(ac_count)

@numba.experimental.jitclass
class ByteBuffer:
    _next_byte: numba.uint64    # type: ignore
    _bytebuffer: numba.uint8[:] # type: ignore
    _bitbuffer: numba.uint64    # type: ignore
    _bitcapacity: numba.uint8   # type: ignore

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
def huffman_encode_block(bytebuffer: ByteBuffer, block: np.ndarray, expanded_dc_table: np.ndarray, expanded_ac_table: np.ndarray):
    def add_dc_codeword(value):
        length, codeword = expanded_dc_table[value]
        bytebuffer.add_bits(length, codeword)

    def add_ac_codeword(value):
        length, codeword = expanded_ac_table[value]
        bytebuffer.add_bits(length, codeword)

    eob = block[-1] == 0
    block = np.trim_zeros(block, trim='b')
    if block.size == 0:
        add_dc_codeword(0)
        add_ac_codeword(EOB)
        return
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

@numba.jit(cache=True)
def huffman_encode(zz_img: np.ndarray, dc_table: Table, ac_table: Table) -> np.ndarray:
    expanded_dc_table = expand_huffman_table(*dc_table)
    expanded_ac_table = expand_huffman_table(*ac_table)
    bytebuffer = ByteBuffer()
    for block in zz_img:
        huffman_encode_block(bytebuffer, block, expanded_dc_table, expanded_ac_table)
    return bytebuffer.final_flush()

@numba.jit(cache=True)
def huffman_encode_color(zz_y: np.ndarray, zz_cb: np.ndarray, zz_cr: np.ndarray, luma_dc: Table, luma_ac: Table, chroma_dc: Table, chroma_ac: Table) -> np.ndarray:
    expanded_luma_dc = expand_huffman_table(*luma_dc)
    expanded_luma_ac = expand_huffman_table(*luma_ac)
    expanded_chroma_dc = expand_huffman_table(*chroma_dc)
    expanded_chroma_ac = expand_huffman_table(*chroma_ac)
    bytebuffer = ByteBuffer()
    for i in range(zz_y.shape[0]):
        huffman_encode_block(bytebuffer, zz_y[i], expanded_luma_dc, expanded_luma_ac)
        huffman_encode_block(bytebuffer, zz_cb[i], expanded_chroma_dc, expanded_chroma_ac)
        huffman_encode_block(bytebuffer, zz_cr[i], expanded_chroma_dc, expanded_chroma_ac)
    return bytebuffer.final_flush()

def lossy_compress(blocks: np.ndarray, q: np.ndarray):
    dct = dctn(blocks, s=(8, 8), axes=(2, 3), norm="ortho")
    dct /= q
    return np.round(dct, out=dct)

def lossy_uncompress(blocks: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = idctn(blocks * q, s=(8, 8), axes=(2, 3), norm="ortho")
    result = cast(np.ndarray, result) # weird type hint bug with scipy's idctn
    return np.round(result, out=result)

def lossy_compress_color_blocks(img: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray):
    ycbcr = np.trunc(img @ RGB_TO_YCBCR).astype(np.int16)
    ycbcr[:, :, :, :, 0] -= 128
    ycbcr[:, :, :, :, 0] = lossy_compress(ycbcr[:, :, :, :, 0], q_luma)
    ycbcr[:, :, :, :, 1] = lossy_compress(ycbcr[:, :, :, :, 1], q_chroma)
    ycbcr[:, :, :, :, 2] = lossy_compress(ycbcr[:, :, :, :, 2], q_chroma)
    return ycbcr

def lossy_compress_color(img: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray):
    return lossy_compress_color_blocks(to_blocks(img), q_luma, q_chroma)

def lossy_uncompress_color_blocks(ycbcr: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray):
    ycbcr[:, :, :, :, 0] = lossy_uncompress(ycbcr[:, :, :, :, 0], q_luma)
    ycbcr[:, :, :, :, 0] += 128
    ycbcr[:, :, :, :, 1] = lossy_uncompress(ycbcr[:, :, :, :, 1], q_chroma)
    ycbcr[:, :, :, :, 2] = lossy_uncompress(ycbcr[:, :, :, :, 2], q_chroma)
    img = (ycbcr @ YCBCR_TO_RGB).round()
    img = np.round(img, out=img)
    img = np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)

def lossy_uncompress_color(img: np.ndarray, q_luma: np.ndarray, q_chroma: np.ndarray, height: int, width: int):
    return from_blocks(lossy_uncompress_color_blocks(img, q_luma, q_chroma), height, width)

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

def to_file(file: BinaryIO, data: np.ndarray, expected_dtype:Any=np.uint8):
    assert data.flags.c_contiguous
    assert data.strides == (data.itemsize,)
    assert data.shape == (data.size,)
    assert data.dtype == expected_dtype
    file.write(memoryview(data))

def write_huffman_table(file: BinaryIO, huffman_lengths: np.ndarray, huffman_values: np.ndarray, is_ac: bool, table_id: int):
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
    to_file(file, huffman_lengths[1:])
    to_file(file, huffman_values)

def write_quantization_matrix(file: BinaryIO, q: np.ndarray, matrix_id: int):
    assert q.shape == (8, 8)
    assert q.itemsize == 2

    extended = 1
    expected_dtype = np.uint16
    if q.max() < 256:
        extended = 0
        expected_dtype = np.uint8
        q = q.astype(np.uint8)

    file.write(
        b"\xFF\xDB" + # DQT
        int(2 + 1 + 64 * q.itemsize).to_bytes(2, 'big') +
        ((extended << 4) + matrix_id).to_bytes(1)
    )

    assert byteorder in ['little', 'big']
    if extended and (byteorder == 'little'):
        # Extended precision matrix needs endianness swap.
        q = q.byteswap()

    # !! Matrix is stored to DQT segment in zigzag order !!
    q = q.reshape(64)[ZIGZAG_IDX]

    to_file(file, q, expected_dtype)

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

def write_grayscale_jpeg(file: BinaryIO, height: int, width: int, q_luma: np.ndarray, luma_dc: Table, luma_ac: Table, entropy_coded_data: np.ndarray):
    file.write(b"\xFF\xD8") # SOI
    write_app0(file)
    write_quantization_matrix(file, q_luma, 0)
    write_start_of_frame(file, height, width, 8, [SOFComponentParameters(1, 1, 1, 0)])
    write_huffman_table(file, *luma_dc, False, 0)
    write_huffman_table(file, *luma_ac, True, 0)
    write_start_of_scan(file, [SOSComponentParameters(1, 0, 0)])
    to_file(file, entropy_coded_data)
    file.write(b"\xFF\xD9") # EOI

def write_color_jpeg(file: BinaryIO, height: int, width: int, q_luma: np.ndarray, q_chroma: np.ndarray, luma_dc: Table, luma_ac: Table, chroma_dc: Table, chroma_ac: Table, entropy_coded_data: np.ndarray):
    file.write(b"\xFF\xD8") # SOI
    write_app0(file)
    write_quantization_matrix(file, q_luma, 0)
    write_quantization_matrix(file, q_chroma, 1)
    write_start_of_frame(file, height, width, 8, [SOFComponentParameters(1, 1, 1, 0), SOFComponentParameters(2, 1, 1, 1), SOFComponentParameters(3, 1, 1, 1)])
    write_huffman_table(file, *luma_dc, False, 0)
    write_huffman_table(file, *luma_ac, True, 0)
    write_huffman_table(file, *chroma_dc, False, 1)
    write_huffman_table(file, *chroma_ac, True, 1)
    write_start_of_scan(file, [SOSComponentParameters(1, 0, 0), SOSComponentParameters(2, 1, 1), SOSComponentParameters(3, 1, 1)])
    to_file(file, entropy_coded_data)
    file.write(b"\xFF\xD9") # EOI

def compress_with_quality(destination: BinaryIO, img: np.ndarray, quality: int):
    q_luma = Q_quality(Q_LUMA, quality)
    img_comp = lossy_compress_grayscale(img, q_luma)
    dc_delta_encode(img_comp)
    zz_img = zigzag(img_comp)
    tables = build_huffman_tables(zz_img)
    entropy_coded_data = huffman_encode(zz_img, *tables)
    write_grayscale_jpeg(destination, img.shape[0], img.shape[1], q_luma, *tables, entropy_coded_data)

def compress_with_quality_color(destination: BinaryIO, img: np.ndarray, quality: int):
    q_luma = Q_quality(Q_LUMA, quality)
    q_chroma = Q_quality(Q_CHROMA, quality)
    img_comp = lossy_compress_color(img, q_luma, q_chroma)
    dc_delta_encode_color(img_comp)
    zz_y = zigzag(img_comp[:, :, :, :, 0])
    zz_cb = zigzag(img_comp[:, :, :, :, 1])
    zz_cr = zigzag(img_comp[:, :, :, :, 2])
    luma_tables = build_huffman_tables(zz_y)
    chroma_tables = build_huffman_tables(zz_cb, zz_cr)
    entropy_coded_data = huffman_encode_color(zz_y, zz_cb, zz_cr, *luma_tables, *chroma_tables)
    write_color_jpeg(destination, img.shape[0], img.shape[1], q_luma, q_chroma, *luma_tables, *chroma_tables, entropy_coded_data)

def lossy_roundtrip_mse_grayscale(img, quality):
    q_luma = Q_quality(Q_LUMA, quality)
    img_comp = lossy_compress_grayscale(img, q_luma)
    img_roundtripped = lossy_uncompress_grayscale(img_comp, q_luma, img.shape)
    return mse(img, img_roundtripped)

def lossy_roundtrip_mse_color(img, quality):
    q_luma = Q_quality(Q_LUMA, quality)
    q_chroma = Q_quality(Q_CHROMA, quality)
    img_comp = lossy_compress_color(img, q_luma, q_chroma)
    img_roundtripped = lossy_uncompress_color(img_comp, q_luma, q_chroma, img.shape[0], img.shape[1])
    return mse(img, img_roundtripped)

def lossy_roundtrip_mse_color_blocks(blocks, quality):
    q_luma = Q_quality(Q_LUMA, quality)
    q_chroma = Q_quality(Q_CHROMA, quality)
    blocks = blocks[np.newaxis, ...]
    img_comp = lossy_compress_color_blocks(blocks, q_luma, q_chroma)
    img_roundtripped = lossy_uncompress_color_blocks(img_comp, q_luma, q_chroma)
    return mse(blocks, img_roundtripped)

def search_with_mse(img: np.ndarray, max_mse: float, roundtrip_fn) -> int:
    l, r = 1, 100
    while l < r:
        m = (l + r) // 2
        m_mse = roundtrip_fn(img, m)
        if m_mse <= max_mse:
            r = m
        else:
            l = m + 1
    return l

def search_with_mse_timed(img: np.ndarray, max_mse: float, roundtrip_fn) -> int:
    start = perf_counter()
    quality = search_with_mse(img, max_mse, roundtrip_fn)
    print(f"MSE search {roundtrip_fn.__name__} took {perf_counter() - start:.2f} s")
    return quality

def pick_random_blocks(img: np.ndarray, block_count: int, random: np.random.Generator):
    blocks = np.empty((block_count, 8, 8, 3), dtype=np.uint8)
    x_coords = random.integers(0, (img.shape[1] - 7) // 8, block_count)
    y_coords = random.integers(0, (img.shape[0] - 7) // 8, block_count)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        blocks[i] = img[y * 8:y * 8 + 8, x * 8:x * 8 + 8]
    return blocks

AVIF_HASINDEX = 0x10
AVIIF_KEYFRAME = 0x10
def compress_video_with_mse(source: Path, destination: BinaryIO, max_mse: float | None, quality: int | None):
    # FFmpegReader uses outdated numpy types
    np.float = np.float64 # type: ignore
    np.int = np.int_      # type: ignore
    reader = FFmpegReader(source)

    def w(data: bytes):
        destination.write(data)
    def w16(data):
        assert 0 <= data < 2**16
        w(int(data).to_bytes(2, 'little'))
    def w32(data):
        assert 0 <= data < 2**32
        w(int(data).to_bytes(4, 'little'))
    def placeholder():
        placeholder_offset = destination.tell()
        w32(0)
        return placeholder_offset
    def fill_placeholder(placeholder_offset: int, data: int):
        old_offset = destination.tell()
        destination.seek(placeholder_offset)
        w32(data)
        destination.seek(old_offset)
    @contextmanager
    def chunk(marker: str, expected_size: int | None = None):
        assert len(marker) == 4
        w(marker.encode('ascii'))
        size_offset = placeholder()
        yield
        chunk_size = destination.tell() - size_offset - 4
        assert (expected_size is None) or (chunk_size == expected_size)
        fill_placeholder(size_offset, chunk_size)
        if chunk_size % 2 == 1:
            w(b"\0")
    @contextmanager
    def list_chunk(marker: str):
        with chunk("LIST"):
            w(marker.encode('ascii'))
            yield

    w(b"RIFF")
    riff_size = placeholder()
    max_byterate_offset = None
    buf_size_1 = None
    buf_size_2 = None
    w(b"AVI ")
    with list_chunk("hdrl"):
        with chunk("avih", 56):
            w32(int(1_000_000 / reader.inputfps)) # microseconds per frame
            max_byterate_offset = placeholder()
            w32(0) # reserved
            w32(AVIF_HASINDEX)
            w32(reader.inputframenum) # total frames
            w32(0) # initial frame
            w32(1) # number of streams
            buf_size_1 = placeholder()
            w32(reader.inputwidth)
            w32(reader.inputheight)
            w(b"\0" * 16) # reserved
        with list_chunk("strl"):
            with chunk("strh", 56):
                w(b"vids")
                w(b"MJPG")
                w32(0) # flags
                w32(0) # priority
                w32(0) # initial frame
                second, frames = reader.probeInfo["video"]["@r_frame_rate"].split("/")
                w32(int(frames))
                w32(int(second))
                w32(0) # start time
                w32(reader.inputframenum) # length
                buf_size_2 = placeholder()
                w(b"\xFF\xFF\xFF\xFF") # quality (not used)
                w32(0) # variable sample size
                w16(0) # left coordinate
                w16(0) # top coordinate
                w16(reader.inputwidth) # right coordinate
                w16(reader.inputheight) # bottom coordinate
            with chunk("strf", 40):
                w32(40) # duplicate size field for some reason
                w32(reader.inputwidth)
                w32(reader.inputheight)
                w16(1) # planes must be 1
                w16(24) # bits per pixel
                w(b"MJPG")
                w32(reader.inputwidth * reader.inputheight * 3) # image size in bytes
                w32(0) # no horizontal DPI
                w32(0) # no vertical DPI
                w32(0) # all colors used
                w32(0) # all colors important

    offsets = np.empty(reader.inputframenum, dtype=np.int32)
    sizes = np.empty(reader.inputframenum, dtype=np.int32)
    random = np.random.default_rng(seed=0)
    with list_chunk("movi"):
        for i, frame in enumerate(reader.nextFrame()):
            offsets[i] = destination.tell()
            w(b"00dc")
            frame_size_offset = placeholder()
            frame_quality = quality
            if frame_quality is None:
                assert max_mse is not None
                block_samples = pick_random_blocks(frame, 64, random)
                frame_quality = search_with_mse(block_samples, max_mse, lossy_roundtrip_mse_color_blocks)
            compress_with_quality_color(destination, frame, frame_quality)
            print(f"Frame {i + 1}/{reader.inputframenum} compressed with quality {frame_quality}")
            frame_size = destination.tell() - frame_size_offset - 4
            fill_placeholder(frame_size_offset, frame_size)
            sizes[i] = frame_size
            if destination.tell() % 2 == 1:
                w(b"\0")

    with chunk("idx1"):
        for i in range(reader.inputframenum):
            w(b"00dc")
            w32(AVIIF_KEYFRAME)
            w32(offsets[i])
            w32(sizes[i])

    fps = int(np.ceil(reader.inputfps))
    sliding_window = np.lib.stride_tricks.sliding_window_view(sizes, fps)
    max_byterate = sliding_window.sum(axis=1).max() + 8
    max_buffer = sizes.max() + 8
    fill_placeholder(max_byterate_offset, max_byterate)
    fill_placeholder(buf_size_1, max_buffer)
    fill_placeholder(buf_size_2, max_buffer)
    fill_placeholder(riff_size, destination.tell() - 8)

def encode_image(img: np.ndarray, output: Path, quality: int, overwrite: bool):
    if len(img.shape) == 2:
        kind = "gray"
        compress = compress_with_quality
    else:
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        kind = "color"
        compress = compress_with_quality_color

    start = perf_counter()
    with open(output, "wb" if overwrite else "xb") as dest:
        compress(dest, img, quality)
    end = perf_counter()

    print(f"Quality {quality: >3} {kind: >5}: {end - start:.2f} s")

def encode_video(video: Path, output: Path, overwrite: bool, max_mse: float | None, quality: int | None):
    assert (max_mse is None) ^ (quality is None)
    with open(output, "wb" if overwrite else "xb") as dest:
        compress_video_with_mse(video, dest, max_mse, quality)
    src_reader = FFmpegReader(video)
    out_reader = FFmpegReader(output)
    assert src_reader.inputframenum == out_reader.inputframenum
    frame_mse = np.empty(src_reader.inputframenum, dtype=np.float64)
    for i, (src_frame, out_frame) in enumerate(zip(src_reader.nextFrame(), out_reader.nextFrame())):
        frame_mse[i] = mse(src_frame, out_frame)
    print(f"Video MSE: {frame_mse.mean():.2f}")

def examples():
    test_blocks_reversible()
    test_huffman_table()

    video_source = Path("Green Bird Perched on Tree Branch.mp4")
    video_output = video_source.with_suffix(".avi")
    if not video_output.exists():
        encode_video(video_source, video_output, overwrite=True, max_mse=15, quality=None)

    img_grayscale = face(gray=True)
    img_color = face(gray=False)
    report = []
    settings = list(product([5, 20, 50, 70, 95, 100], [(img_grayscale, "gray"), (img_color, "color")]))
    max_mse = 15
    settings.append((search_with_mse_timed(img_grayscale, max_mse, lossy_roundtrip_mse_grayscale), (img_grayscale, "gray")))
    settings.append((search_with_mse_timed(img_color, max_mse, lossy_roundtrip_mse_color), (img_color, "color")))

    for quality, (img, kind) in settings:
        destination = Path(f"quality={quality}_{kind}.jpg")
        encode_image(img, destination, quality, overwrite=True)

        size = destination.stat().st_size
        img_roundtripped = imread(destination)
        jpg_mse = mse(img, img_roundtripped)
        jpg_snr = snr(img, img_roundtripped)
        report.append(f"Quality {quality: >3} {kind: >5}: MSE = {jpg_mse: >6.2f}, SNR = {jpg_snr: >9.2f}, Size = {size // 1024: >6} KiB")

    Path("report.txt").write_text("\n".join(report) + "\n", encoding="ascii")

def main():
    ap = ArgumentParser()
    ap.add_argument("COMMAND", choices=["image", "video", "examples"], default="examples", nargs="?")
    ap.add_argument("SOURCE", type=Path, nargs="?")
    ap.add_argument("-o", type=Path)
    ap.add_argument("--quality", type=int)
    ap.add_argument("--mse", type=float)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.COMMAND == "examples":
        examples()
        return

    source = args.SOURCE
    assert source is not None
    output = args.o
    if output is None:
        output = source.with_suffix(".jpg" if args.COMMAND == "image" else ".avi")

    if args.COMMAND == "image":
        img = imread(source)
        quality = args.quality
        if quality is None:
            assert args.mse is not None
            roundtrip = lossy_roundtrip_mse_grayscale if len(img.shape) == 2 else lossy_roundtrip_mse_color
            quality = search_with_mse_timed(img, args.mse, roundtrip)
        encode_image(img, output, quality, args.overwrite)
    elif args.COMMAND == "video":
        encode_video(source, output, args.overwrite, args.mse, args.quality)

if __name__ == "__main__":
    main()
