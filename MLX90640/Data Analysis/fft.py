import numpy as np
from scipy import fftpack, signal

# Load the thermal image as a 2D numpy array
thermal_image = np.load(r"C:\Users\animo\Downloads\Annotation\data\1_person_2\3154.967049826.npy")
thermal_image = thermal_image[100]

# Apply the Fourier transform to the image
fft_image = fftpack.fft2(thermal_image)

# Calculate the power spectrum
power_spectrum = np.abs(fft_image) ** 2

# Calculate the dominant frequency
freqs = fftpack.fftfreq(thermal_image.shape[0]) * 1.0 / (thermal_image.shape[1] * (1.0 / np.mean(np.diff(np.arange(thermal_image.shape[0])))))
power_spectrum_1D = np.sum(power_spectrum, axis=1)
dominant_frequency = np.abs(freqs[np.argmax(power_spectrum_1D)])

# Apply the continuous wavelet transform to the image
scales = np.arange(1, 10)
wavelet = signal.cwt(thermal_image, signal.ricker, scales)

# Calculate the wavelet coefficients
wavelet_coeffs = np.abs(wavelet)

print("Dominant frequency: ", dominant_frequency)
print("Power spectrum: ", power_spectrum)
print("Wavelet coefficients: ", wavelet_coeffs)

#%%

# Apply Fourier transform to the image
f = np.fft.fft2(thermal_image)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Calculate the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Extract the dominant frequency and power spectrum
rows, cols = thermal_image.shape
crow, ccol = rows//2, cols//2
dominant_frequency = np.argmax(np.abs(fshift[crow, :])) / cols
power_spectrum = np.sum(magnitude_spectrum**2) / (rows * cols)

print('Dominant Frequency:', dominant_frequency)
print('Power Spectrum:', power_spectrum)

#%%

