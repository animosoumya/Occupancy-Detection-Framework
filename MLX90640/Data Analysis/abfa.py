#%% Import required libraries

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

#%% provide path for thermal data directory

path = r"C:\Users\animo\Downloads\Annotation\data"
os.chdir(path)

#%% Trying new variables for adv_blob_filt_algorithm

high_threshold = 0 # Similar to peak detection algorithm
num_centroid   = 0 
loc_centroid   = 0 

#%% old variables used in ABFA

smin                = 5      # Minimum size of the blob
smax                = 15     # Maximum Size of the blob
delta_b             = 0.05   # Parameter to change the threshold value
feature_list        = []
valid_blobs         = []

#%% Calculate threshold frame from bground frames 

def get_threshold_frame(bg_frames):
    
    thresh_frame = 0
    # init_thresh_frame = 0
    
    for bg_frame in bg_frames:
        thresh_frame += bg_frame
    
    thresh_frame = thresh_frame/len(bg_frames)
    
    delta_a = 4 * np.std(thresh_frame)
    
    thresh_frame = thresh_frame + delta_a
    
    # init_thresh_frame = thresh_frame.copy()
    
    return thresh_frame

#%% Extract subpage 0 and subpage 1 from frames

def create_subpage(frames):
    
    subpage_list_0  = []
    subpage_list_1  = []
    
    for frame in frames:
    
        subpage_0 = np.zeros((24,32))
        subpage_1 = np.zeros((24,32))
        
        frame = frame.reshape(24,32)
        
        for i in range(0,24):
            for j in range(0,32):
                if ((i%2 == 0 and j%2 !=0) or (i%2 != 0 and j%2 ==0)):
                    subpage_0[i][j] = frame[i][j]
                else:
                    subpage_1[i][j] = frame[i][j]
        
        subpage_list_0.append(subpage_0)
        subpage_list_1.append(subpage_1)
    
    return subpage_list_0, subpage_list_1

#%% Reduce noise by using bilinear interpolation

def noise_reduction(subpage_list_0, subpage_list_1):
    
    conv_subpage_0  = subpage_list_0.copy()
    conv_subpage_1  = subpage_list_1.copy()
    denoised_frames = []
    
    for i in range(0, len(subpage_list_0)):
        
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
                    
        denoised_frames.append(conv_subpage_0[i] + conv_subpage_1[i])
    
    return denoised_frames

#%% Binarize the data frame into black and white

def binarize(data_frame,t_frame):
    
    binary_frame  = data_frame.copy()
    num_active_pixels = 0
    
    for k in range(0,24):
        
        for l in range(0,32):
            
            if (binary_frame[k][l] <= t_frame[k][l]):
                binary_frame[k][l]  = 0
            
            else:
                binary_frame[k][l]  = 255
                num_active_pixels += 1
                
    return binary_frame, num_active_pixels

#%% define the blob extraction function

def blob_extraction(data_frame,thresh_frame):
    
    binary_frame, active_pixels = binarize(data_frame, thresh_frame)
    
    img = np.uint8(binary_frame)
    
    cca_output = cv2.connectedComponentsWithStatsWithAlgorithm(img, 8, cv2.CV_32S, cv2.CV_32S)
    
    return cca_output

#%% define get_mask function to get data frame cooresponding to a mask

def get_mask(blobs, data_frame, i):    
    
    int_mask = (blobs == i).astype("uint8") * 255
    mask     = int_mask.copy()
    mask     = mask.astype('float64')
    
    for i in range(0,24):
        for j in range (0,32):
            if mask[i][j] == 255:
                mask[i][j] = data_frame[i][j]
                
    return mask

#%% Define blob filtering algorithm

def blob_filtering(data_frame, thresh_frame):    
    
    output = blob_extraction(data_frame, thresh_frame)
    num_blobs, blobs, blob_data, centroids = output
    
    for i in range(1, num_blobs):
        
        if blob_data[i,4] < smin :
            
            continue
        
        elif blob_data[i,4] > smax :
            
            mask = get_mask(blobs, data_frame, i)
            blob_filtering(mask, thresh_frame + delta_b)
            
        else:
            
            valid_blobs.append(get_mask(blobs, data_frame, i))  
            
    return valid_blobs

#%% Example

bg_frames = np.load(r"C:\Users\animo\Downloads\Annotation\data\0_person_2\3296.064467186.npy")

thresh_frame = 0
for bg_frame in bg_frames:
    thresh_frame += bg_frame

thresh_frame = thresh_frame/len(bg_frames)

delta_a = 4 * np.std(thresh_frame)

thresh_frame = thresh_frame + delta_a

#%% Get centriod features

def get_centroid_data(data_frame, thresh_frame):
    
    thermal_image = np.load(r"C:\Users\animo\Downloads\Annotation\data\2_person_2\3019.240416668.npy")
    data_frame = thermal_image[100]
    
    cca_output     = blob_extraction(data_frame, thresh_frame)
    filtered_blobs = blob_filtering(data_frame, thresh_frame)
    centroid_loc   = cca_output[3]
    # centroid_min = 
    # centroid_max = 
    
    return centroid_loc

#%% Extract features from data_frame

def feature_extraction(denoised_frames, thresh_frame, gt):
    
    for data_frame in denoised_frames:
        
        global valid_blobs
        
        mean  = np.mean(data_frame - thresh_frame)
        sigma = np.std (data_frame - thresh_frame)
        
        binary_frame, num_active_pixels = binarize(data_frame, thresh_frame)
        
        num_conn_comp = len(blob_filtering(data_frame, thresh_frame))
        
        valid_blobs = []
        
        centroid_data = get_centroid_data(data_frame, thresh_frame)
        
        feature_list.append([mean, sigma, num_active_pixels, num_conn_comp, centroid_data, gt])
    
    return

#%% Get background frames for current sub_path

def get_bg_frames (sub_path, gt_path):
    
    gt_path = '0' + gt_path[1:]
    bg_path = os.path.join(os.path.dirname(sub_path), gt_path)
    
    for file in os.listdir(bg_path):
        
        if file.endswith(".npy"):
            
            file = os.path.join(bg_path, file)
            bg_frames = np.load(file, allow_pickle = True)
    
    return bg_frames

#%% append background features to feature_list

def extract_bg_features(bg_frames):
            
    for bg_frame in bg_frames:
        
        thresh_frame = get_threshold_frame(bg_frames)
        
        mean  = np.mean(bg_frame - thresh_frame)
        sigma = np.std (bg_frame - thresh_frame)
        
        binary_frame, num_active_pixels = binarize(bg_frame, thresh_frame)
        
        feature_list.append([mean, sigma, num_active_pixels, 0, 0, 0, 0])

#%% main() function

for sub_path, dirs, file_list in os.walk(path):
    
    for file in file_list:
        
        gt_path = os.path.basename(sub_path).split('/')[-1]
        gt = int (gt_path[0]) # first character of the sub_paths
        
        if file.endswith(".npy"):
            
            if gt == 0:

                bg_frames = get_bg_frames(sub_path, gt_path)
                extract_bg_features(bg_frames)
            
            else:

                bg_frames    = get_bg_frames(sub_path, gt_path)          
                thresh_frame = get_threshold_frame(bg_frames)
                
                file_path    = os.path.join(sub_path, file)
                pixel_frames = np.load(file_path, allow_pickle = True)
          
                subpage_list_0, subpage_list_1 = create_subpage(pixel_frames)
                
                denoised_frames = noise_reduction(subpage_list_0, subpage_list_1)
        
                feature_extraction(denoised_frames, thresh_frame, gt)
            

data = pd.DataFrame(feature_list, columns = ['mean', 'sigma', 'num_active_pixels', 'num_conn_comp', 'gt'])
feature_path = os.path.join(os.path.dirname(path), "features.csv")
data.to_csv(feature_path)