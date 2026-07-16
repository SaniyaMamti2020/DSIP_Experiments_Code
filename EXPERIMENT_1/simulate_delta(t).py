import numpy as np
import matplotlib.pyplot as plt

def signal(t):
    y = np.zeros_like(t)

    y[t == 0] = 1
    y[t == 1] += 1
    y[t == -5] += 3

    return y

t = np.arange(-10,11)

y = signal(t)

plt.stem(t, y)
plt.title("y(t)=δ(t)+δ(t-1)+3δ(t+5)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()