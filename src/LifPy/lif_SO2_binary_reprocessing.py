'''

lif_SO2_binary_reprocessing.py

Run this script after generating the processing_variables.txt file using 
lif_NO_generate_processing_variables.py
Details of file structure and how to run this code are found in a README file 
in the LIF-Reprocessing-Tools github.

'''

import os

import pandas as pd
import lif_functions as lif    
 
data_dir = (r'C:\Users\pp835\OneDrive - University of York\Documents\Data Analysis\CARES\SO2_code_test')

day_folders = lif.find_day_folders(data_dir)

channel_count = 10  # 10 = single channel, 11 = dual channel

channel_format ={
    'sig_counts': 0
    , 'ref_counts': 1
    , 'seed_LD_current': 2
    , 'laser_pwr_PT0': [3, 4]
    , 'time_ms': [7, 8]
    , 'seed_LD_mode': 9
}  

HK_headers_dict = ['Task', 'Cell_Flow', 'Cal_SO2_MFC_set', 'Cal_SO2_MFC_Read']

#processing_variables = pd.read_csv(os.path.join(data_dir, 'processing_variables.txt'))

HK_data = lif.import_HK_data(data_dir, day_folders)

#for i in processing_variables.index:
    
lif.reprocess_binary_data(
    data_freq = 100 # can take 10 or 100
    , date = '20250602'
    , file = 'LIFCnts_20250602 1.txt'
    , log_start_datetime = '31/05/2025 16:06:41'
    , data_dir = data_dir
    , HK_data = HK_data
    , channel_count = channel_count
    , channel_format = channel_format 
    , HK_headers_dict = HK_headers_dict
    , histograms = True
)