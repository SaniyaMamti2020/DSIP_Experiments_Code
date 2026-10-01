import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_median_filter(image, kernel_size): 
    # Apply median filter to remove noise
    filtered_image = cv2.medianBlur(image, kernel_size) 
    return filtered_image

def apply_bilateral_filter(image, d, sigma_color, sigma_space): 
    # Apply bilateral filter to remove noise
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space) 
    return filtered_image

# Load the input image 
image_path = 'input_image.png'
input_image = cv2.imread(image_path)

# Apply median filter
median_filtered_image = apply_median_filter(input_image, kernel_size=5)

# Apply bilateral filter
bilateral_filtered_image = apply_bilateral_filter(input_image,d=9,sigma_color=75,sigma_space=75)

# Display the original image and the filtered images side by side
combined_image = np.hstack((input_image,median_filtered_image, bilateral_filtered_image))
cv2.imshow("Original | Median Filtered | Bilateral Filtered", combined_image) 
cv2.waitKey(0)
 

# Save the filtered images (optional) 
median_filtered_path = 'median_filtered_image.jpg' 
bilateral_filtered_path = 'bilateral_filtered_image.jpg'
cv2.imwrite(median_filtered_path, median_filtered_image) 
cv2.imwrite(bilateral_filtered_path, bilateral_filtered_image) 
print(f"Median filtered image saved at: {median_filtered_path}") 
print(f"Bilateral filtered image saved at: {bilateral_filtered_path}")