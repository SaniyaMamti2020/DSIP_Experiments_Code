import numpy as np
import matplotlib.pyplot as plt

def continuous_ramp(t, slope):
    y = np.zeros_like(t)
    y[t >= 0] = slope * t[t >= 0]
    return y

def discrete_ramp(samples, slope):
    y = np.zeros(samples)
    y[samples // 2:] = slope * np.arange(samples // 2, samples)
    return y

t = np.linspace(-5,5,1000)
slope = 2
samples = 20

continuous = continuous_ramp(t, slope)
discrete = discrete_ramp(samples, slope)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, continuous)
plt.title("Continuous Ramp")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(discrete)
plt.title("Discrete Ramp")
plt.grid(True)

plt.tight_layout()
plt.show()