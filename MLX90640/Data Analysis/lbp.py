import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage import io

# Load the thermal image
thermal_image = np.load(r"C:\Users\animo\Downloads\Annotation\data\1_person_2\3154.967049826.npy")
thermal_image = thermal_image[100]

# Extract LBP features
radius = 5
n_points = 8 * radius
lbp = local_binary_pattern(thermal_image, n_points, radius, method='uniform')

# Display the LBP image
plt.imshow(lbp, cmap='gray')
plt.axis('off')
plt.show()