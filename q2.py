import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.io import imread
from skimage.color import rgb2gray

# Load and preprocess image
im = imread("Neural.JPG")
img = rgb2gray(im) * 255
plt.imshow(img, cmap='gray')
plt.show()

# Crop image
img1 = img[40:350, 20:350]
plt.imshow(img1, cmap='gray')
plt.show()

# Define filters
fil1 = np.array([[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]])

fil2 = np.array([[0.2, 0.5, 0.2],
                 [0.5, 1, 0.5],
                 [0.2, 0.5, 0.2]])

fil3 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1, 0.1, 0.1]])

# Apply convolution
grad1 = signal.convolve2d(img1, fil1, boundary='symm', mode='same')
grad2 = signal.convolve2d(img1, fil2, boundary='symm', mode='same')
grad3 = signal.convolve2d(img1, fil3, boundary='symm', mode='same')

# Display results
plt.imshow(abs(grad1), cmap='gray', vmin=np.min(grad1), vmax=np.max(grad1))
plt.show()
plt.imshow(grad2, cmap='gray')
plt.show()
plt.imshow(grad3, cmap='gray')
plt.show()
