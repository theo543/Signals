import numpy as np
import sounddevice as sd

def main():
    sample_rate = 48_000
    time = np.arange(0, 1, 1 / sample_rate)
    a = np.sin(time * 2 * np.pi * 400)
    b = np.sin(time * 2 * np.pi * 800)
    c = np.append(a, b)
    sd.play(c, samplerate=sample_rate, blocking=True)
    # Putting the signals in the same vector doesn't change anything, they're still clearly different.

if __name__ == "__main__":
    main()
