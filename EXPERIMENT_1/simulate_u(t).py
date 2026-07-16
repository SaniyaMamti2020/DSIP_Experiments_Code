import numpy as np
import matplotlib.pyplot as plt

def signal(t):
    y = np.zeros_like(t)

    y[t >= 0] = 1
    y[t >= 1] += 1
    y[t >= -5] += 3

    return y

t = np.linspace(-10,10,1000)

y = signal(t)

plt.plot(t, y)
plt.title("y(t)=u(t)+u(t-1)+3u(t+5)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()