import cv2
import numpy as np
import matplotlib.pyplot as plt

# Calculate HSV histogram
def get_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

# Calculate color moments
def get_color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv, axis=(0, 1))
    std = np.std(hsv, axis=(0, 1))
    skew = np.mean((hsv - mean) ** 3, axis=(0, 1))
    return np.concatenate([mean, std, skew])

# Calculate RGB histogram
def get_rgb_histogram(img):
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
    cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])

# Feature extraction
def extract_features(img):
    hsv_hist = get_hsv_histogram(img)
    color_moments = get_color_moments(img)
    rgb_hist = get_rgb_histogram(img)
    return np.concatenate([hsv_hist, color_moments, rgb_hist])

# Plot HSV histogram
def plot_hsv_histogram(hist):
    plt.figure()
    plt.title("HSV Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.plot(hist, color='b')
    plt.xlim([0, 180])
    plt.show()

# Plot color moments
def plot_color_moments(moments):
    labels = ['Mean H', 'Mean S', 'Mean V', 'Std H', 'Std S', 'Std V', 'Skew H', 'Skew S', 'Skew V']
    plt.figure()
    plt.title("Color Moments")
    plt.bar(labels, moments, color='g')
    plt.xlabel("Moments")
    plt.ylabel("Value")
    plt.show()