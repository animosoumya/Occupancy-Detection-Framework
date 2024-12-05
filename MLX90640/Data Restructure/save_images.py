#%%

# Analyse Images of cases where occupancy is more than 12

#%% Import required libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

#%% provide path to thermal images

pixel_matrices = []
numpyfile = TemporaryFile()

path = r"F:\Documents\MS\Pi_4B\MLX\data\person_4\data_15"

for file in os.listdir(path):
        
    if file.endswith(".npy"):
        
        file_path    = os.path.join(path, file)
        pixel_matrix = np.load(file_path, allow_pickle= True)
        pixel_matrix = pixel_matrix[0, :, :]
        pixel_matrices.append(pixel_matrix)
    
pixel_matrices = np.stack(pixel_matrices)
numpyfile = str(file)
np.save(r"F:\Documents\MS\Pi_4B\MLX\data\person_4\data_15" + numpyfile, pixel_matrices, allow_pickle= True)
pixel_matrices = []

#%% save images in folder 

i = 0
pixel_matrices = np.load(r"F:\Documents\MS\Pi_4B\MLX\data\person_4\data_15Images.npy")

for matrix in pixel_matrices:
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(16,12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(matrix)
    plt.savefig(r"F:\Documents\MS\Pi_4B\MLX\data\person_4\data_15\Images\image-" + str(i))
    i += 1
    plt.close()