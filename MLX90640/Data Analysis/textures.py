import numpy as np
from skimage.feature import greycomatrix, greycoprops


# Load the thermal image as a 2D numpy array
thermal_image = np.load(r"C:\Users\animo\Downloads\Annotation\data\2_person_2\3019.240416668.npy")
thermal_image = thermal_image[100].astype("uint8")

bg_frames = np.load(r"C:\Users\animo\Downloads\Annotation\data\0_person_2\.npy")


thresh_frame = 0
for bg_frame in bg_frames:
    thresh_frame += bg_frame

thresh_frame = thresh_frame/len(bg_frames)

delta_a = 4 * np.std(thresh_frame)

thresh_frame = thresh_frame + delta_a

# Define the distance and angle offsets for calculating co-occurrence matrix
d = 1
theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Calculate the co-occurrence matrix
glcm = greycomatrix(thermal_image, distances=[d], angles=theta, levels=256, symmetric=True, normed=True)

# Calculate the contrast, correlation, energy, and homogeneity features from the co-occurrence matrix
contrast = greycoprops(glcm, 'contrast')
correlation = greycoprops(glcm, 'correlation')
energy = greycoprops(glcm, 'energy')
homogeneity = greycoprops(glcm, 'homogeneity')

# Concatenate the features into a single vector
texture_features = np.concatenate((contrast.flatten(), correlation.flatten(), energy.flatten(), homogeneity.flatten()))

print(texture_features)