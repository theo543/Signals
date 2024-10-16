import tempfile
import numpy as np
import sounddevice as sd
import scipy

def sawtooth(t):
    return (t - np.floor(t)) * 2 - 1

def square(t):
    return np.mod(np.floor(t), 2) * 2 - 1

def main():
    sample_rate = 48_000
    time = np.arange(0, 1, 1 / sample_rate)
    a = np.sin(time * 2 * np.pi * 400)
    with tempfile.TemporaryFile() as temp_file:
        # test WAV writing and reading
        scipy.io.wavfile.write(temp_file, sample_rate, a)
        orig_a = a
        read_rate, a = scipy.io.wavfile.read(temp_file)
        assert read_rate == sample_rate
        assert np.all(orig_a == a)
    b = np.sin(time * 2 * np.pi * 800)
    c = sawtooth(time * 2 * np.pi * 240)
    d = square(time * 2 * np.pi * 300)
    for s in [a, b, c, d]:
        sd.play(s, blocking=True, samplerate=sample_rate)

if __name__ == "__main__":
    main()
