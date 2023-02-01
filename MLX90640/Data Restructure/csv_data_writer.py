from fnmatch import fnmatch
import numpy as np
import pandas as pd
import csv
import os


root = r'C:\Users\animo\Downloads\Annotation\data'


pattern_1 = "*.npy"
pattern_2 = "*.csv"

for path, subdirs, files in os.walk(root):
    
    data_arrays = []
    df = pd.DataFrame(columns = ['timestamp', 'pixel_values'])
    
    for name in files:
        
        if fnmatch(name, pattern_1):
            
            file_name   = os.path.join(path, name)
            data_arrays = np.load(file_name)
            data_arrays = data_arrays.transpose(0,1,2).reshape(150,-1)
        
        if fnmatch(name, pattern_2) and name != "ir.csv":
            file_name = os.path.join(path, name)
            ts = pd.read_csv(file_name)
            df['timestamp'] = ts['0']
            df2 = {'timestamp': df.iloc[-1][0] + 1, 'pixel_values': None}
            df = df.append(df2, ignore_index = True)
        
    fieldnames = ['no', 'timestamp', 'data_array']    
    file_name = os.path.join(path, "ir.csv")
    
    if len(data_arrays) > 0:
        with open(file_name, 'w', newline = '' ) as csvfile:
            
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fieldnames)

            for i in range (150):
                data_str = ','.join([f"{t:.2f}" for t in data_arrays[i]])
                data_row = [i, df['timestamp'].iloc[i], data_str]
                csvwriter.writerow(data_row)
