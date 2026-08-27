import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Generate histogram
def generate_histogram(image):
    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )
    return histogram

# Histogram Equalization
def perform_histogram_equalization(image):

    # Convert image to grayscale
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    return equalized_image

# Histogram Matching
def perform_histogram_matching(input_image, reference_image):

    # Convert images to grayscale
    gray_input = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_reference = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate histograms
    input_hist = cv2.calcHist(
        [gray_input],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    reference_hist = cv2.calcHist(
        [gray_reference],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    # Calculate CDF
    input_cdf = np.cumsum(input_hist)
    reference_cdf = np.cumsum(reference_hist)

    # Normalize CDF
    input_cdf = input_cdf / input_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]

    # Create mapping
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        difference = np.abs(
            reference_cdf - input_cdf[i]
        )

        mapping[i] = np.argmin(difference)

    # Apply mapping
    matched_image = mapping[gray_input]

    return matched_image

# Load Input Image

input_path = 'input_image.png'

input_image = cv2.imread(input_path)

if input_image is None:
    print("Error: Input image not found!")
    print("Check:", input_path)
    exit()

# Input Histogram

gray_input = cv2.cvtColor(
    input_image,
    cv2.COLOR_BGR2GRAY
)

input_hist = generate_histogram(gray_input)

plt.plot(input_hist)
plt.title("Input Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# Histogram Equalization

equalized_image = perform_histogram_equalization(
    input_image
)

equalized_hist = generate_histogram(
    equalized_image
)

plt.plot(equalized_hist)
plt.title("Equalized Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# Load Reference Image

reference_path = 'reference_image.png'

reference_image = cv2.imread(
    reference_path
)

if reference_image is None:
    print("Error: Reference image not found!")
    print("Check:", reference_path)
    exit()

# Histogram Matching

matched_image = perform_histogram_matching(
    input_image,
    reference_image
)

# Matched Histogram

matched_hist = generate_histogram(
    matched_image
)

plt.plot(matched_hist)
plt.title("Matched Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# Convert grayscale images to BGR

equalized_color = cv2.cvtColor(
    equalized_image,
    cv2.COLOR_GRAY2BGR
)

matched_color = cv2.cvtColor(
    matched_image,
    cv2.COLOR_GRAY2BGR
)

# Display Images

combined_image = np.hstack(
    (
        input_image,
        equalized_color,
        matched_color
    )
)

cv2.imshow(
    "Original | Equalized | Matched",
    combined_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Images

equalized_path = 'equalized_image.jpg'

matched_path = 'matched_image.jpg'

cv2.imwrite(
    equalized_path,
    equalized_image
)

cv2.imwrite(
    matched_path,
    matched_image
)

print(
    f"Equalized image saved at: {equalized_path}"
)

print(
    f"Matched image saved at: {matched_path}"
)