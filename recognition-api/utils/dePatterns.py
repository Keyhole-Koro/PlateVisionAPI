import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_clusters(image, k=3, output_path="cluster_plot.png"):
    h, w, _ = image.shape
    reshaped = image.reshape((-1, 3)).astype(np.float32)

    # K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = np.uint8(centers)

    # Plot color centers
    plt.figure()
    for i, center in enumerate(centers):
        plt.bar(i, 1, color=(center[2]/255, center[1]/255, center[0]/255))

    # Print color ranges
    for cluster_idx in range(k):
        cluster_pixels = reshaped[labels == cluster_idx]
        r_min, g_min, b_min = cluster_pixels.min(axis=0)
        r_max, g_max, b_max = cluster_pixels.max(axis=0)
        print(f"Cluster {cluster_idx}: R=({r_min},{r_max}), G=({g_min},{g_max}), B=({b_min},{b_max})")

    plt.title("K-Means Cluster Centers")
    plt.xticks(range(k), [f"Cluster {i}" for i in range(k)])
    
    # Save the plot as an image
    plt.savefig(output_path)
    plt.close()
    
def apply_kmeans(image, k=3, blur_strength=5):
    """Separate colors in the image using K-Means clustering and set the background to white"""
    h, w, _ = image.shape  # Get the image dimensions
    pixels = h * w  # Total number of pixels

    # Apply Gaussian blur with adjustable strength
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    reshaped = blurred.reshape((-1, 3))  # Reshape to a 2D array of (H*W, 3)
    reshaped = np.float32(reshaped)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Flatten the labels array
    labels = labels.flatten()

    # Set the brightest cluster as the background
    centers = np.uint8(centers)
    brightness = np.sum(centers, axis=1)  # Calculate the brightness of each cluster
    background_idx = np.argmax(brightness)  # Set the brightest cluster as the background

    # Convert the background cluster to white
    mask = labels.reshape((h, w)) == background_idx
    output = image.copy()
    output[mask] = [255, 255, 255]  # Replace with white

    # Extract clusters close to dark green
    dark_green = np.array([0, 100, 0])  # Dark green in BGR
    distances = np.linalg.norm(centers - dark_green, axis=1)
    dark_green_idx = np.argmin(distances)

    # Remove other clusters
    mask = labels.reshape((h, w)) != dark_green_idx
    output[mask] = [255, 255, 255]  # Replace with white

    return output

def dePatterns(image, blur_strength=5):
    """Remove patterns from the license plate image, leaving only the text"""
    # Set the background to white using K-Means (keep only the dark green cluster)
    cleaned_plate = apply_kmeans(image, k=3, blur_strength=blur_strength)

    return cleaned_plate
