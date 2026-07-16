import numpy as np
import matplotlib.pyplot as plt

def continuous_step(t):
    y = np.zeros_like(t)
    y[t >= 0] = 1
    return y

def discrete_step(samples):
    y = np.zeros(samples)
    y[samples // 2:] = 1
    return y

t = np.linspace(-5, 5, 1000)
continuous = continuous_step(t)

samples = 20
discrete = discrete_step(samples)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, continuous)
plt.title("Continuous Unit Step")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(discrete)
plt.title("Discrete Unit Step")
plt.grid(True)

plt.tight_layout()
plt.show()