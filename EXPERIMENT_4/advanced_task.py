
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, bilinear, freqz, lfilter

# 1. Signal and filter specifications
fs = 16000 # Sampling frequency in Hz
duration = 0.05 # Duration in seconds
frequencies = [1000, 2000, 3000, 4000]
filter_order = 4
ripple = 0.5 # Chebyshev passband ripple in dB

t = np.arange(0, duration, 1/fs)
N = len(t)

# 2. Generate four discrete sine waves
x1 = np.sin(2 * np.pi * 1000 * t)
x2 = np.sin(2 * np.pi * 2000 * t)
x3 = np.sin(2 * np.pi * 3000 * t)
x4 = np.sin(2 * np.pi * 4000 * t)

# Plot individual sine waves in time domain
plt.figure(figsize=(12, 8))

for i, (signal, f) in enumerate(zip([x1, x2, x3, x4], frequencies), 1):
    plt.subplot(2, 2, i)
    plt.plot(t * 1000, signal)
    plt.title(f'Sine Wave: {f/1000} kHz')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 3)
    plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Add all four signals
x = x1 + x2 + x3 + x4

# Plot combined signal in time domain
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, x)
plt.title('Sum of Four Sine Waves - Time Domain')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. FFT function for frequency-domain analysis
def plot_spectrum(signal, title):
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), d=1/fs)

    magnitude = np.abs(spectrum) / len(signal)

    if len(signal) % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2

    plt.figure(figsize=(10, 4))
    plt.stem(freq / 1000, magnitude)
    plt.title(title)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Frequency spectrum of combined signal
plot_spectrum(x, 'Combined Signal - Frequency Domain')

# 5. Design Butterworth filter using bilinear method
fc_butter = 2000  # Cutoff frequency in Hz

# Prewarp cutoff frequency for bilinear transformation
Omega_butter = 2 * fs * np.tan(np.pi * fc_butter / fs)
analog_b, analog_a = butter(filter_order, Omega_butter,analog=True, btype='low')
digital_b, digital_a = bilinear(analog_b, analog_a, fs=fs)

# Filter combined signal
y_butter = lfilter(digital_b, digital_a, x)

# Butterworth output: time domain
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, y_butter)
plt.title('Butterworth Output - Time Domain (2 kHz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

# Butterworth output: frequency domain
plot_spectrum(y_butter,'Butterworth Output - Frequency Domain')

# Butterworth filter magnitude response
f_b, h_b = freqz(digital_b, digital_a, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(f_b / 1000, 20*np.log10(np.maximum(np.abs(h_b), 1e-10)))
plt.axvline(2, color='r', linestyle='--',label='Cutoff = 2 kHz')
plt.title('Butterworth Filter Frequency Response')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 8)
plt.ylim(-100, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Design Chebyshev Type I using bilinear method
fc_cheby = 3000  # Cutoff frequency in Hz

Omega_cheby = 2 * fs * np.tan(np.pi * fc_cheby / fs)
analog_b, analog_a = cheby1(filter_order, ripple, Omega_cheby,analog=True, btype='low')
digital_b, digital_a = bilinear(analog_b, analog_a, fs=fs)

# Filter combined signal
y_cheby = lfilter(digital_b, digital_a, x)

# Chebyshev output: time domain
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, y_cheby)
plt.title('Chebyshev Output - Time Domain (3 kHz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

# Chebyshev output: frequency domain
plot_spectrum(y_cheby,'Chebyshev Output - Frequency Domain')

# Chebyshev filter magnitude response
f_c, h_c = freqz(digital_b, digital_a, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(f_c / 1000, 20*np.log10(np.maximum(np.abs(h_c), 1e-10)))
plt.axvline(3, color='r', linestyle='--', label='Cutoff = 3 kHz')
plt.title('Chebyshev Filter Frequency Response')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 8)
plt.ylim(-100, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Save digital filter coefficients
np.savetxt('butterworth_coefficients.txt',np.vstack((digital_b, digital_a)),delimiter=',')

print('Signal generation and filtering completed.')
print('Filter coefficient file saved.')