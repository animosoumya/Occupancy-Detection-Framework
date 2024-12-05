import numpy as np
import pandas as pd

data_arrays = np.load("C:\\Users\\animo\\Downloads\\MS\\CNN\\my_data\\1436.594498436.npy")

data_arrays = data_arrays.transpose(0,1,2).reshape(200,-1)

df = pd.DataFrame(columns = ['timestamp', 'pixel_values'])

ts = pd.read_csv("C:\\Users\\animo\\Downloads\\MS\\CNN\\my_data\\1436.594498436.csv")
ts  = ts.drop('Unnamed: 0', '0')

df['timestamp'] = ts['0']

i = 0
for array in data_arrays:
    pixel_list = []
    pixel_list = array
    df.iloc[i,1] = pixel_list
    i += 1