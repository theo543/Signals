from matplotlib import pyplot as plt
import numpy as np
from det_svg import savefig

def draw_sampled(ax, fn, t, sampling_freq, drawing_freq, display_fraction = 1.0):
    sampling_range = np.linspace(0, t, num=int(t*sampling_freq))
    sampled_fn = fn(sampling_range)
    drawing_range = np.linspace(0, t, num=int(t*drawing_freq))
    drawn_fn = fn(drawing_range)
    display = int(len(drawing_range) * display_fraction)
    ax.plot(drawing_range[0:display], drawn_fn[0:display])
    display = int(len(sampling_range) * display_fraction)
    ax.stem(sampling_range[0:display], sampled_fn[0:display])

def draw(ax, fn, t, freq, display_fraction = 1.0):
    range_ = np.linspace(0, t, num=int(t * freq))
    samples = fn(range_)
    display_range = int(len(range_) * display_fraction)
    ax.plot(range_[0:display_range], samples[0:display_range])

def signal(fn, freq, phase):
    return lambda t : fn(freq * np.pi * t + phase)

def cos_signal(freq, phase):
    return signal(np.cos, freq, phase)

def sin_signal(freq, phase):
    return signal(np.sin, freq, phase)

def sawtooth(t):
    return (t - np.floor(t)) * 2 - 1

def square(t):
    return np.mod(np.floor(t), 2) * 2 - 1

def spiral(res):
    half_res = res / 2
    x = np.zeros((res, res)) + np.arange(0, res)
    y = np.zeros((res, res)) + np.arange(0, res)[:, np.newaxis]
    distance = np.sqrt((half_res - x) ** 2 + (half_res - y) ** 2) / (half_res * np.sqrt(2))
    slowing_down_distance = - ((1 - distance) ** 2)  * (0.28 * distance - 0.2) / 0.2
    expected_degrees = sawtooth(slowing_down_distance * 10)
    degrees = np.arctan2(y - half_res, x - half_res) / np.pi
    spilling_spiral = np.abs(1 - np.abs(degrees - expected_degrees))
    spiral_ = spilling_spiral * np.minimum(1, 1.6 - distance) ** 10
    return spiral_

def main():
    functions = [cos_signal(520, np.pi / 3), cos_signal(280, -np.pi / 3), cos_signal(120, np.pi / 3)]
    _, axs = plt.subplots(len(functions))
    T = 0.03
    SAMPLING_FREQ = 200
    DRAWING_FREQ = 20000
    for ax, fn in zip(axs, functions):
        draw_sampled(ax, fn, T, SAMPLING_FREQ, DRAWING_FREQ)
    savefig("Lab 1 - Ex 1.svg")
    plt.close()

    _, axs = plt.subplots(4)
    draw_sampled(axs[0], sin_signal(400, 0), 1, 1600, DRAWING_FREQ, display_fraction=0.01)
    draw(axs[1], sin_signal(800, 0), 3, DRAWING_FREQ, display_fraction=0.01)
    draw(axs[2], signal(sawtooth, 240, 0), 0.05, DRAWING_FREQ)
    draw(axs[3], signal(square, 300, 0), 0.05, DRAWING_FREQ)
    savefig("Lab 1 - Ex 2 A-D.svg")
    plt.close()

    rand = np.random.rand(128, 128)
    plt.imshow(rand)
    savefig("Lab 1 - Ex 2 E.svg")
    plt.close()

    low_res_spiral = spiral(128)
    plt.imshow(low_res_spiral)
    savefig("Lab 1 - Ex 2 F.svg")
    plt.close()

    high_res_spiral = spiral(1000)
    plt.imshow(high_res_spiral)
    savefig("Lab 1 - Ex 2 F (high-res).svg")
    plt.close()

    # ex 3
    # (a) interval = 1 / 2000 = 0.0005
    # (b) bytes = 60 * 60 * 2000 * 4 / 8 = 3_600_000

if __name__ == "__main__":
    main()
