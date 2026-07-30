import numpy as np
import matplotlib.pyplot as plt

# Cross-Correlation Function
def cross_correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')

# Autocorrelation Function
def autocorrelation(signal):
    return np.correlate(signal, signal, mode='full')

# Input Signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

# Compute Correlations
cross_corr = cross_correlation(signal1, signal2)
auto_corr = autocorrelation(signal1)

# X-axis (0 to 8)
lags = np.arange(len(cross_corr))

# Print Results
print("Signal 1:", signal1)
print("Signal 2:", signal2)

print("\nCross-Correlation:")
print(cross_corr)

print("\nAutocorrelation:")
print(auto_corr)

# Plot Graphs
plt.figure(figsize=(10, 6))

# Cross-Correlation
plt.subplot(2, 1, 1)
plt.stem(lags, cross_corr, linefmt='C0-', markerfmt='C0o', basefmt='r-')
plt.title("Cross-correlation")
plt.xlabel("Time Lag")
plt.ylabel("Magnitude")
plt.grid(False)

# Autocorrelation
plt.subplot(2, 1, 2)
plt.stem(lags, auto_corr, linefmt='C0-', markerfmt='C0o', basefmt='r-')
plt.title("Autocorrelation")
plt.xlabel("Time Lag")
plt.ylabel("Magnitude")
plt.grid(False)

plt.tight_layout()
plt.show()

# Save Results
np.savetxt("cross_correlation.txt", cross_corr, fmt="%d")
np.savetxt("autocorrelation.txt", auto_corr, fmt="%d")

print("\nResults saved successfully!")