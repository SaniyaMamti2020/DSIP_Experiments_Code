import numpy as np
import matplotlib.pyplot as plt

# Function to design FIR filter
def design_fir_filter(cutoff_freq, filter_length, window_type):
    # Calculate the middle point
    M = filter_length - 1
    n = np.arange(filter_length)
    # Ideal low-pass filter using sinc function
    h = 2 * cutoff_freq * np.sinc(2 * cutoff_freq * (n - M / 2))
    # Apply window
    if window_type == 'hamming':window = np.hamming(filter_length)
    elif window_type == 'hann':window = np.hanning(filter_length)
    elif window_type == 'blackman':window = np.blackman(filter_length)
    else:window = np.ones(filter_length)
    # FIR filter coefficients
    filter_coefficients = h * window
    # Normalize coefficients
    filter_coefficients = filter_coefficients / np.sum(filter_coefficients)
    return filter_coefficients

def plot_filter_response(filter_coefficients):
    # Calculate frequency response using FFT
    N = 2048
    frequency_response = np.fft.fft(filter_coefficients,N)
    # Take only first half of frequency response
    frequency_response = frequency_response[:N // 2]
    # Normalized frequency
    frequency = np.linspace(0,0.5,N // 2)
    # Magnitude response
    magnitude = np.abs(frequency_response)
    # Convert magnitude into dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    # Normalize magnitude so maximum is 0 dB
    magnitude_db = magnitude_db - np.max(magnitude_db)
    # Magnitude Response
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequency,magnitude_db)
    plt.title('FIR Filter Magnitude Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim(0, 0.5)
    plt.ylim(-100, 10)
    plt.show()
    # Impulse Response
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(filter_coefficients)),filter_coefficients)
    plt.title('FIR Filter Impulse Response')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
# Filter Specifications
cutoff_frequency = 0.2
filter_length = 51
window_type = 'hamming'
# Design FIR Filter
filter_coefficients = design_fir_filter(cutoff_frequency,filter_length,window_type)
# Display Filter Coefficients
print("FIR Filter Coefficients:")
print(filter_coefficients)
# Plot Filter Response
plot_filter_response(filter_coefficients)
# Save Filter Coefficients
filter_path = 'fir_filter_coefficients.txt'
np.savetxt(filter_path,filter_coefficients,delimiter=',')
print(f"Filter coefficients saved at: {filter_path}")