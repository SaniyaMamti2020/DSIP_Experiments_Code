import numpy as np
import matplotlib.pyplot as plt

# Input signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Compute FFT
fft_result = np.fft.fft(signal)

# Signal length
signal_length = np.arange(len(signal))

# Magnitude spectrum
magnitude_spectrum = np.abs(fft_result)

# Phase spectrum
phase_spectrum = np.angle(fft_result)

# Compute IFFT
reconstructed_signal = np.fft.ifft(fft_result)

# Display FFT values
print("FFT Result:")
print(fft_result)

print("\nMagnitude Spectrum:")
print(magnitude_spectrum)

print("\nPhase Spectrum:")
print(phase_spectrum)

print("\nReconstructed Signal:")
print(reconstructed_signal)

# Plot
plt.figure(figsize=(12, 6))

# Original signal
plt.subplot(2, 1, 1)
plt.stem(signal_length, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Magnitude spectrum
plt.subplot(2, 1, 2)
plt.stem(signal_length, magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()