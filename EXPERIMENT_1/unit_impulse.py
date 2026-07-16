import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(length, position):
    signal = np.zeros(length)
    signal[position] = 1
    return signal

# Parameters
start = -10
stop = 10
step = 1

x = np.arange(start, stop + step, step)

impulse_signal = unit_impulse(len(x), abs(start) // step)

plt.stem(x, impulse_signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Unit Impulse Signal")
plt.grid(True)
plt.show()