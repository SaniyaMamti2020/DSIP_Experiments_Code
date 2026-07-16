import numpy as np
import matplotlib.pyplot as plt

def continuous_parabolic(t, coeff):
    return np.polyval(coeff, t)

def discrete_parabolic(samples, coeff):
    n = np.arange(samples)
    return np.polyval(coeff, n)

t = np.linspace(-5,5,1000)

coeff = [1,2,1]
samples = 20

continuous = continuous_parabolic(t, coeff)
discrete = discrete_parabolic(samples, coeff)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, continuous)
plt.title("Continuous Parabolic")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(discrete)
plt.title("Discrete Parabolic")
plt.grid(True)

plt.tight_layout()
plt.show()