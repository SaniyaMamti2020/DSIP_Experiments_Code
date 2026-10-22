
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_gaussian_filter(image, sigma):
    # Convert image to float32
    image = np.float32(image)

    # Perform Fourier Transform
    frequency_domain = cv2.dft(
        image, flags=cv2.DFT_COMPLEX_OUTPUT
    )

    # Shift zero-frequency component to the center
    shifted_frequency_domain = np.fft.fftshift(
        frequency_domain, axes=(0, 1)
    )

    # Create Gaussian filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    x = np.arange(rows) - crow
    y = np.arange(cols) - ccol
    X, Y = np.meshgrid(y, x)

    gaussian = np.exp(
        -(X**2 + Y**2) / (2 * sigma**2)
    )

    # Create a two-channel mask
    mask = gaussian[:, :, np.newaxis]

    # Apply Gaussian filter
    filtered_frequency_domain = (
        shifted_frequency_domain * mask
    )

    # Shift zero-frequency component back
    shifted_filtered_frequency_domain = np.fft.ifftshift(
        filtered_frequency_domain, axes=(0, 1)
    )

    # Inverse Fourier Transform
    filtered_image = cv2.idft(
        shifted_filtered_frequency_domain,
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

    # Convert image to uint8
    filtered_image = np.clip(
        filtered_image, 0, 255
    ).astype(np.uint8)

    return filtered_image


def apply_sharpening_filter(image, strength):
    # Convert image to float32
    image = np.float32(image)

    # Perform Fourier Transform
    frequency_domain = cv2.dft(
        image, flags=cv2.DFT_COMPLEX_OUTPUT
    )

    # Shift zero-frequency component to the center
    shifted_frequency_domain = np.fft.fftshift(
        frequency_domain, axes=(0, 1)
    )

    # Create high-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.float32)

    # Remove low frequencies from the center
    mask[
        max(0, crow - strength):min(rows, crow + strength),
        max(0, ccol - strength):min(cols, ccol + strength)
    ] = 0

    # Apply high-pass filter
    filtered_frequency_domain = (
        shifted_frequency_domain * mask[:, :, np.newaxis]
    )

    # Shift frequency component back
    shifted_filtered_frequency_domain = np.fft.ifftshift(
        filtered_frequency_domain, axes=(0, 1)
    )

    # Inverse Fourier Transform
    high_frequency_image = cv2.idft(
        shifted_filtered_frequency_domain,
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

    # Add high-frequency details to the original image
    sharpened_image = image + high_frequency_image

    # Convert to uint8
    sharpened_image = np.clip(
        sharpened_image, 0, 255
    ).astype(np.uint8)

    return sharpened_image

# Load input image in grayscale
image_path = 'image.png'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check whether the image was loaded
if input_image is None:
    raise FileNotFoundError(
        f"Image not found. Please check the path: {image_path}"
    )

# Apply Gaussian smoothing
sigma = 20
smoothed_image = apply_gaussian_filter(
    input_image, sigma
)

# Apply sharpening
strength = 20
sharpened_image = apply_sharpening_filter(
    input_image, strength
)

# Display images side by side
combined_image = np.hstack((
    input_image,
    smoothed_image,
    sharpened_image
))

plt.figure(figsize=(15, 5))
plt.imshow(combined_image, cmap='gray')
plt.title("Original | Smoothed | Sharpened")
plt.axis('off')
plt.show()

# Save filtered images
smoothed_path = 'smoothed_image.jpg'
sharpened_path = 'sharpened_image.jpg'

cv2.imwrite(smoothed_path, smoothed_image)
cv2.imwrite(sharpened_path, sharpened_image)

print(f"Smoothed image saved at: {smoothed_path}")
print(f"Sharpened image saved at: {sharpened_path}")