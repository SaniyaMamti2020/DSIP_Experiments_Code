import numpy as np
import matplotlib.pyplot as plt

def continuous_sine(t, A, f, phase):
    return A * np.sin(2 * np.pi * f * t + phase)

def discrete_sine(samples, Fs, A, f, phase):
    t = np.arange(samples) / Fs
    return A * np.sin(2 * np.pi * f * t + phase)

t = np.linspace(0,1,1000)

A = 1
f = 2
phase = 0

Fs = 10
samples = 100

continuous = continuous_sine(t, A, f, phase)
discrete = discrete_sine(samples, Fs, A, f, phase)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, continuous)
plt.title("Continuous Sine Wave")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(discrete)
plt.title("Discrete Sine Wave")
plt.grid(True)

plt.tight_layout()
plt.show()