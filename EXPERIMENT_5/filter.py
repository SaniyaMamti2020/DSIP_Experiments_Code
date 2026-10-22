import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
import os

# FIR FILTER DESIGN USING HAMMING WINDOW
# Sampling Frequency = 10 kHz
# Number of Coefficients N = 31

FS = 10000          # Sampling frequency in Hz
N = 31              # Number of filter coefficients (taps)
WINDOW = "hamming"

OUTPUT_DIR = "FIR_Filter_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NYQUIST = FS / 2

print("=" * 60)
print("FIR FILTER DESIGN")
print("=" * 60)
print(f"Sampling Frequency : {FS} Hz")
print(f"Nyquist Frequency  : {NYQUIST} Hz")
print(f"Filter Length (N)  : {N}")
print(f"Window Type        : {WINDOW.capitalize()}")


# Function to design and display an FIR filter

def design_fir_filter(name, cutoff, filter_type):
    """
    Design FIR filters using the Hamming window.
    cutoff is given in Hz.
    """

    coefficients = firwin(
        numtaps=N,
        cutoff=cutoff,
        window=WINDOW,
        pass_zero=filter_type,
        fs=FS
    )

    # Print coefficients
    print(f"\n{'=' * 60}")
    print(f"{name.upper()} FILTER")
    print("=" * 60)
    print("Filter coefficients:")
    for i, value in enumerate(coefficients):
        print(f"h[{i:02d}] = {value:.8f}")

    # Save coefficients
    filename = os.path.join(
        OUTPUT_DIR,
        name.lower().replace(" ", "_") + "_coefficients.txt"
    )
    np.savetxt(filename, coefficients, fmt="%.10f")
    print(f"\nCoefficients saved to: {filename}")

    # Calculate frequency response
    frequencies, response = freqz(
        coefficients,
        worN=4096,
        fs=FS
    )

    magnitude = np.abs(response)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Plot magnitude response
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, magnitude_db)
    plt.title(f"{name} FIR Filter - Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, NYQUIST)
    plt.ylim(-100, 10)
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        name.lower().replace(" ", "_") + "_magnitude.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # Plot impulse response
    plt.figure(figsize=(10, 5))
    plt.stem(np.arange(N), coefficients)
    plt.title(f"{name} FIR Filter - Impulse Response")
    plt.xlabel("Sample Number (n)")
    plt.ylabel("Amplitude h[n]")
    plt.grid(True)
    plt.tight_layout()

    impulse_path = os.path.join(
        OUTPUT_DIR,
        name.lower().replace(" ", "_") + "_impulse.png"
    )
    plt.savefig(impulse_path, dpi=150)
    plt.show()

    return coefficients

# 1. LOW-PASS FILTER
# Cutoff frequency = 3 kHz

lowpass = design_fir_filter(name="Low Pass",cutoff=3000,filter_type="lowpass")

# 2. HIGH-PASS FILTER
# Cutoff frequency = 3 kHz

highpass = design_fir_filter(name="High Pass",cutoff=3000,filter_type="highpass")

# 3. BAND-PASS FILTER
# Requested cutoff frequencies = 3 kHz and 6 kHz
# IMPORTANT:
# Nyquist frequency = 5 kHz, so 6 kHz is invalid.
# Use 4.9 kHz as the practical upper design edge.
# This is an approximation, NOT a true 3-6 kHz filter.

print("\nWARNING: Band-pass upper cutoff of 6000 Hz is")
print("above the 5000 Hz Nyquist frequency.")
print("Designing an approximate 3000-4900 Hz band-pass filter.")

bandpass = design_fir_filter(
    name="Band Pass",
    cutoff=[3000, 4900],
    filter_type="bandpass"
)

# 4. BAND-STOP FILTER
# Requested cutoff frequencies = 3 kHz and 6 kHz
# The same Nyquist limitation applies.

print("\nWARNING: Band-stop upper cutoff of 6000 Hz is")
print("above the 5000 Hz Nyquist frequency.")
print("Designing an approximate 3000-4900 Hz band-stop filter.")

bandstop = design_fir_filter(name="Band Stop",cutoff=[3000, 4900],filter_type="bandstop")

# SAVE ALL FOUR FILTERS IN ONE FILE
all_coefficients = np.column_stack((lowpass, highpass, bandpass, bandstop))
combined_path = os.path.join(OUTPUT_DIR,"all_fir_filter_coefficients.txt")
np.savetxt(combined_path,all_coefficients,delimiter=",",fmt="%.10f",header="LowPass,HighPass,BandPass_3to4.9kHz,"
           "BandStop_3to4.9kHz",comments="")

print("\n" + "=" * 60)
print("ALL FILTERS DESIGNED SUCCESSFULLY")
print("=" * 60)
print(f"Results folder: {OUTPUT_DIR}")
print(f"Combined coefficients: {combined_path}")
print("Magnitude and impulse-response plots have been saved.")
print("\nNote: The requested 3-6 kHz filters require")
print("a sampling frequency greater than 12 kHz.")