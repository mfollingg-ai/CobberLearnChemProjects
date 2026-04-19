import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# =========================================================
# 1. CREATE IMAGE (5x5)
# =========================================================
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

print("INPUT IMAGE:")
print(image)

# =========================================================
# 2. FILTER (HORIZONTAL EDGE DETECTOR)
# =========================================================
kernel = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

print("\nFILTER:")
print(kernel)

# =========================================================
# 3. APPLY CONVOLUTION
# =========================================================
feature_map = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

print("\nFEATURE MAP:")
print(feature_map)

# =========================================================
# 4. ANALYSIS
# =========================================================
print("\nMAX VALUE:", np.max(feature_map))
print("MIN VALUE:", np.min(feature_map))

# =========================================================
# 5. VISUALIZATION (IMPORTANT PART)
# =========================================================
plt.figure(figsize=(10, 4))

# -------------------------
# INPUT IMAGE
# -------------------------
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Input Image (5x5)")
plt.xticks([])
plt.yticks([])

# -------------------------
# FEATURE MAP
# -------------------------
plt.subplot(1, 2, 2)
plt.imshow(feature_map, cmap="coolwarm")
plt.title("Feature Map (After Convolution)")
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.tight_layout()
plt.show()