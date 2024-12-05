# import the required libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

# Folder Path for background pixels
backg_matrix   = np.load(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_4\data_0\3296.064467186.npy", allow_pickle=True)
backg_matrices = np.split(backg_matrix, 200, axis=0)

# Folder Path for active pixels
path = r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_4\Mixed_pixels"
os.chdir(path)
pixel_matrices = []

for file in os.listdir():
    
    if file.endswith(".npy"):
        matrix_list = np.load(file, allow_pickle = True)
        pixel_matrices.append(np.split(matrix_list, 200, axis=0))

#%%

# Define variables
smin       = 4      # Minimum size of the blob
smax       = 12     # Maximum Size of the blob
t_frame    = 0      # Threshold frame
it_frame   = 0      # To store the initial threshold frame
delta_a    = 1.5    # Threshold Temperature value
num_img    = 1      # number of images
blobs      = 0      # blobs extracted from CCL
blob_data  = 0      # Stats of blobs extracted from CCL
num_blobs  = 0      # total number of blobs extracted 
centroids  = 0      # Centroid of the blobs
delta_b    = 0.15  # Parameter to change the threshold value

#%%

# Find the Threshold Frame
for j in range (0,len(backg_matrices)):
    t_frame  += backg_matrices[j]
    it_frame += backg_matrices[j]

t_frame  = (t_frame/len(backg_matrices)) + delta_a
t_frame  = t_frame.reshape(24,32)
it_frame = (it_frame/len(backg_matrices)) + delta_a
it_frame = it_frame.reshape(24,32)


for i in range(18,24):
    for j in range(0,32):
        t_frame[i][j]  = t_frame[i][j]  - 0.25
    
#%%
#print(3*np.std(it_frame - delta_a))

# Define subpage lists and frames to store all the pixel matrices
subpage_list_0  = []
subpage_list_1  = []
conv_subpage_0  = []
conv_subpage_1  = []
data_frame      = []
binary_frame    = []
empty_list      = []
filtered_frames = []
valid_blobs     = []


#define kernel
kernel = np.array([[0,0.25,0],[0.25,0,0.25],[0,0.25,0]])

#%% 

# Append the values to the lists of subpage and convoluted subpages
for j in range (0,len(pixel_matrices)):
    for i in range(0, len(pixel_matrices[0])):
        
        subpage_0 = np.zeros((24,32))
        subpage_1 = np.zeros((24,32))
        
        a = pixel_matrices[j][i].reshape(24,32)
        
        for l in range(0,24):
            for k in range(0,32):
                if ((l%2 == 0 and k%2 !=0) or (l%2 != 0 and k%2 ==0)):
                    subpage_0[l][k] = a[l][k]
                else:
                    subpage_1[l][k] = a[l][k]
        
        subpage_list_0.append(subpage_0)
        subpage_list_1.append(subpage_1)
    
#%%

# Copy subpage lists into convolution subpages
conv_subpage_0 = subpage_list_0.copy()
conv_subpage_1 = subpage_list_1.copy()



for i in range (0, len(subpage_list_0)):
    padded_0 = np.pad(subpage_list_0[i], ((1, 1), (1, 1)), 'reflect')
    padded_1 = np.pad(subpage_list_1[i], ((1, 1), (1, 1)), 'reflect')
    for j in range(0,24):
        for k in range(0,32):     
            if (subpage_list_0[i][j][k] == 0):
                conv_subpage_0[i][j][k] = (0.25) * (padded_0[j][k+1] + padded_0[j+1][k] + padded_0[j+1][k+2] + padded_0[j+2][k+1])
            else:
                conv_subpage_0[i][j][k] = 0
   
            if (subpage_list_1[i][j][k] == 0):
                conv_subpage_1[i][j][k] = (0.25) * (padded_1[j][k+1] + padded_1[j+1][k] + padded_1[j+1][k+2] + padded_1[j+2][k+1])
            else:
                conv_subpage_1[i][j][k] = 0
    data_frame.append(conv_subpage_0[i] + conv_subpage_1[i])

#%%

# Check difference between normal and interploated image
# test_img = 1513
# plt.imshow(pixel_matrices[7][113].reshape(24,32))
# plt.imshow(subpage_list_0[test_img], cmap='hot')
# plt.imshow(subpage_list_1[test_img])
# plt.imshow(conv_subpage_0[test_img])
# plt.imshow(conv_subpage_1[test_img])
# plt.imshow(data_frame[test_img])

#%%

def faulty_frame(conv_subpage_0, conv_subpage_1):
    delta_m = 3
    delta_s = 0.02 
    mean  = abs(np.mean(conv_subpage_0 - conv_subpage_1))
    sigma = abs(np.std (conv_subpage_0 - conv_subpage_1))
    if (mean < delta_m and sigma < delta_s):
        return 1
    else:
        return 0

#%% binarize the data frame into black and white

def binarize(data_frame,t_frame):
    binary_frame  = data_frame.copy()
    active_pixels = 0
    for k in range(0,24):
        for l in range(0,32):
            if (binary_frame[k][l] <= t_frame[k][l]):
                binary_frame[k][l]  = 0
            else:
                binary_frame[k][l]  = 255
                active_pixels += 1
    return binary_frame, active_pixels

#%% define the blob extraction function

def blob_extraction(data_frame,t_frame):
    
    binary_frame, active_pixels = binarize(data_frame, t_frame)
    img = np.uint8(binary_frame)
    # Coonected Components Labeling
    output = cv2.connectedComponentsWithStatsWithAlgorithm(img, 8, cv2.CV_32S, cv2.CV_32S)
    return output

#%% define get_mask function to get data frame cooresponding to a mask

def get_mask(blobs,data_frame, i):    
    global mask    
    int_mask = (blobs == i).astype("uint8") * 255
    mask     = int_mask.copy()
    mask     = mask.astype('float64')
    for i in range(0,24):
        for j in range (0,32):
            if mask[i][j] == 255:
                mask[i][j] = data_frame[i][j]
    return mask

#%% Define blob filtering algorithm

def blob_filtering(data_frame,t_frame):    
    global delta_b, smax
    global data_frame_2
        
    output = blob_extraction(data_frame,t_frame)
    num_blobs, blobs, blob_data, centroids = output
    
    for i in range(1,num_blobs):
        if blob_data[i,4] < smin :
            continue
        elif blob_data[i,4] > smax :
            # Extract frame using blob as a mask on original frame
            data_frame_2 = get_mask(blobs, data_frame, i)
            blob_filtering(data_frame_2, t_frame + delta_b)
        else:
            # Record blob as valid blob
            valid_blobs.append(get_mask(blobs,data_frame, i))            
    return valid_blobs

#%% Extract features from data frame

def feature_extraction(data_frame, it_frame):
    
    # Distribution of pixels in a frame
    mean  = np.mean(data_frame - it_frame)
    sigma = np.std (data_frame - it_frame)
    
    # Total number of active pixels in a frame is propertional to number of people
    binary_frame, active_pixels = binarize(data_frame, it_frame)
    
    output = blob_extraction(data_frame, it_frame)
    
    num_blobs, blobs, blob_data, centroids = output
     
    return mean, sigma, active_pixels

#%% Add background values to fetaures.csv file with gt=0

for i in range(0, len(backg_matrices)):
    
    filtered_frame  = np.zeros((24,32))
    backg_matrices[i] = backg_matrices[i].reshape(24,32)
    fv1, fv2, fv3 = feature_extraction(backg_matrices[i], it_frame)
    empty_list.append([fv1, fv2, fv3, 0, 0])

#%% Main function that will run for all the data frames

for i in range(0, len(data_frame)):
    
    filtered_frame  = np.zeros((24,32))
    
    valid_blobs = blob_filtering(data_frame[i], t_frame)
    
    for j in range (0, len(valid_blobs)):
        filtered_frame += valid_blobs[j]
        
    filtered_frames.append(filtered_frame) 
    
    # plt.imshow(filtered_frames[0])  
    
    fv1, fv2, fv3 = feature_extraction(data_frame[i], it_frame)

    empty_list.append([fv1, fv2, fv3, len(valid_blobs), 15-(int(i/200))])
    
    valid_blobs = []

#%%

# convert data frae to features.csv file
data = pd.DataFrame(empty_list, columns = ['mean', 'sigma', 'active_pixels', 'cc', 'gt'])
data.to_csv(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_4\frame_features.csv")


plt.imshow(data_frame[815])
plt.imsave(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_4\thermal_img.png", data_frame[15])

#%%
