import numpy as np
import matplotlib.pyplot as plt

# Function for Linear Convolution
def linear_convolution(signal1, signal2):
    return np.convolve(signal1, signal2, mode='full')

# Function for Circular Convolution
def circular_convolution(signal1, signal2):
    fft_length = max(len(signal1), len(signal2))

    fft_signal1 = np.fft.fft(signal1, fft_length)
    fft_signal2 = np.fft.fft(signal2, fft_length)

    return np.fft.ifft(fft_signal1 * fft_signal2)

# Input Signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

# Perform Convolution
linear_conv = linear_convolution(signal1, signal2)
circular_conv = circular_convolution(signal1, signal2)

# Display Results
print("Linear Convolution:")
print(linear_conv)

print("\nCircular Convolution:")
print(np.real(circular_conv))

# Plot Results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(linear_conv)
plt.title("Linear Convolution")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.stem(np.real(circular_conv))
plt.title("Circular Convolution")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()