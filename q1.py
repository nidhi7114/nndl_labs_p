import numpy as np
import matplotlib.pyplot as plt

# Define the signal X and filters H_L, H_H
X = np.array([0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,0,0])
H_L = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
H_H = np.array([-1, 2, -1])

# Perform convolution
y_low = np.convolve(X, H_L, mode='same')  # Low-pass filter
y_high = np.convolve(X, H_H, mode='same')  # High-pass filter

# Plot the results
plt.figure(figsize=(10,5))
plt.plot(X, label='Original Signal', marker='o')
plt.plot(y_low, label='Low-pass Filter Output', linestyle='--', marker='s')
plt.plot(y_high, label='High-pass Filter Output', linestyle='-.', marker='^')

# Labels and legend
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Convolution with Low-Pass and High-Pass Filters')
plt.legend()
plt.grid()
plt.show()
