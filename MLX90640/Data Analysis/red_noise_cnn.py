#%% import the required libraries

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Folder Path for Thermal Data

path = r"C:\Users\animo\Downloads\Annotation\data"
os.chdir(path)

#%% Append the values to the lists of subpage and convoluted subpages

def create_subpage(matrices):
    
    subpage_list_0  = []
    subpage_list_1  = []
    
    for matrix in matrices:
    
        subpage_0 = np.zeros((24,32))
        subpage_1 = np.zeros((24,32))
        
        matrix = matrix.reshape(24,32)
        
        for i in range(0,24):
            for j in range(0,32):
                if ((i%2 == 0 and j%2 !=0) or (i%2 != 0 and j%2 ==0)):
                    subpage_0[i][j] = matrix[i][j]
                else:
                    subpage_1[i][j] = matrix[i][j]
        
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

#%% Write pixel values in ir2.csv files and store them in path directory

def create_ircsv(data_frames, file_path):
    
    df = pd.DataFrame(columns = ['timestamp', 'pixel_values'])
    
    data_frames = np.array(data_frames)
    data_frames = data_frames.transpose(0,1,2).reshape(data_frames.shape[0],-1)
    
    csv_path = file_path.replace(file_path[len(file_path) - 3 : ] , "csv")
    
    ts = pd.read_csv(csv_path)
    df['timestamp'] = ts['0']
    df2 = {'timestamp': df.iloc[-1][0] + 1, 'pixel_values': None}
    df = df.append(df2, ignore_index = True)
    
    fieldnames = ['no', 'timestamp', 'data_array']    
    file_name = os.path.join(os.path.dirname(file_path), "ir2.csv")
    
    
    with open(file_name, 'w', newline = '' ) as csvfile:
        
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fieldnames)

        for i in range (data_frames.shape[0]):
            data_str = ','.join([f"{t:.2f}" for t in data_frames[i]])
            data_row = [i, df['timestamp'].iloc[i], data_str]
            csvwriter.writerow(data_row)
    
    return 

#%% main() function

for path, dirs, file_list in os.walk(path):
        
    for file in file_list:
        
        if file.endswith(".npy"):
            
            file_path = os.path.join(path, file)
            matrix_list = np.load(file_path, allow_pickle = True)
            subpage_list_0, subpage_list_1 = create_subpage(matrix_list)
            denoised_frames = noise_reduction(subpage_list_0, subpage_list_1)
            create_ircsv(denoised_frames, file_path)
            
