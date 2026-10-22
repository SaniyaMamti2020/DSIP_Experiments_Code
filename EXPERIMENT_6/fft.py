import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# FIR FILTER DESIGN + FFT / IFFT IN ONE FILE
# Sampling frequency = 10 kHz; number of taps = 31

Fs = 10_000
N = 31
nyquist = Fs / 2

LPF_CUTOFF = 3000
HPF_CUTOFF = 3000

# 6000 Hz is above Nyquist (5000 Hz), so it cannot be used
# at Fs=10000 Hz. Use 3000-4000 Hz to demonstrate BPF and BSF.
BAND_LOW = 3000
BAND_HIGH = 4000

print("FIR FILTER DESIGN")
print("-----------------")
print(f"Sampling frequency : {Fs} Hz")
print(f"Number of taps     : {N}")
print(f"Nyquist frequency  : {nyquist} Hz")
print("WARNING: Requested 6000 Hz band edge is invalid for Fs=10000 Hz.")
print("Using 3000-4000 Hz for executable BPF/BSF designs.")
print("For a 6000 Hz edge, sampling frequency must be greater than 12000 Hz.")

# Design filters using the windowed-sinc method with a Hamming window.
lpf = firwin(N, LPF_CUTOFF, fs=Fs, window="hamming", pass_zero="lowpass")
hpf = firwin(N, HPF_CUTOFF, fs=Fs, window="hamming", pass_zero="highpass")
bpf = firwin(N, [BAND_LOW, BAND_HIGH], fs=Fs, window="hamming",pass_zero="bandpass")
bsf = firwin(N, [BAND_LOW, BAND_HIGH], fs=Fs, window="hamming",pass_zero="bandstop")

filters = {
    "FIR Low-Pass (3 kHz)": lpf,
    "FIR High-Pass (3 kHz)": hpf,
    f"FIR Band-Pass ({BAND_LOW/1000:.0f}-{BAND_HIGH/1000:.0f} kHz)": bpf,
    f"FIR Band-Stop ({BAND_LOW/1000:.0f}-{BAND_HIGH/1000:.0f} kHz)": bsf,
}

# Display filter coefficients
for name, coeffs in filters.items():
    print(f"\n{name} coefficients:")
    print(np.array2string(coeffs, precision=6, suppress_small=True))

# Plot each filter frequency response separately (one figure per task)
for name, coeffs in filters.items():
    frequencies, response = freqz(coeffs, worN=2048, fs=Fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(response), 1e-10))

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, magnitude_db)
    plt.title(f"{name} - Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, nyquist)
    plt.ylim(-100, 5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot each filter's coefficients separately (one figure per task)
for name, coeffs in filters.items():
    plt.figure(figsize=(10, 5))
    plt.stem(np.arange(N), coeffs)
    plt.title(f"{name} - FIR Coefficients")
    plt.xlabel("Sample Index")
    plt.ylabel("Coefficient Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ORIGINAL FFT / IFFT EXAMPLE
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

fft_result = np.fft.fft(signal)
signal_length = np.arange(len(signal))
magnitude_spectrum = np.abs(fft_result)
phase_spectrum = np.angle(fft_result)
reconstructed_signal = np.fft.ifft(fft_result)

print("\nFFT Result:")
print(fft_result)
print("\nMagnitude Spectrum:")
print(magnitude_spectrum)
print("\nPhase Spectrum (radians):")
print(phase_spectrum)
print("\nReconstructed Signal:")
print(np.real_if_close(reconstructed_signal))

# Separate plots for the original signal, magnitude, phase, and reconstruction
plt.figure(figsize=(9, 4))
plt.stem(signal_length, signal)
plt.title("Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
plt.stem(signal_length, magnitude_spectrum)
plt.title("FFT Magnitude Spectrum")
plt.xlabel("FFT Bin")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
plt.stem(signal_length, phase_spectrum)
plt.title("FFT Phase Spectrum")
plt.xlabel("FFT Bin")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
plt.stem(signal_length, reconstructed_signal.real)
plt.title("Reconstructed Signal Using IFFT")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

if np.allclose(signal, reconstructed_signal.real):
    print("\nIFFT verification: Original signal reconstructed successfully.")
else:
    print("\nIFFT verification: Reconstruction differs from original.")