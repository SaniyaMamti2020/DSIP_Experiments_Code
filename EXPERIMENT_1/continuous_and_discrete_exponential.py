import numpy as np
import matplotlib.pyplot as plt

def continuous_exponential(t, A, b):
    return A * np.exp(b * t)

def discrete_exponential(samples, A, b):
    n = np.arange(samples)
    return A * np.exp(b * n)

t = np.linspace(0,5,1000)

A = 2
b = -0.5
samples = 20

continuous = continuous_exponential(t, A, b)
discrete = discrete_exponential(samples, A, b)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, continuous)
plt.title("Continuous Exponential")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(discrete)
plt.title("Discrete Exponential")
plt.grid(True)

plt.tight_layout()
plt.show()