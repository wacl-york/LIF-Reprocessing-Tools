'''

lif_NO_binary_reprocessing.py

Run this script after generating the processing_variables.txt file using 
lif_NO_generate_processing_variables.py
Details of file structure and how to run this code are found in a README file 
in the LIF-Reprocessing-Tools github.

'''

import os

import pandas as pd
import lif_functions as lif    
 
data_dir = (r'C:\Users\pp835\Documents\PhD Data\COCO-VOC\Processing')

day_folders = lif.find_day_folders(data_dir)

channel_count = 12  # 10 = single channel, 11 = two channel, 12 = three channel

channel_format ={
    'sig_A_counts': 0
    , 'ref_counts': 1
    , 'seed_LD_current': 2
    , 'laser_pwr_PT0': [3, 4]
    , 'time_ms': [7, 8]
    , 'seed_LD_mode': 9
    , 'sig_B_counts': 10
    , 'sig_C_counts': 11
}  

HK_headers_dict = ['Time_s', 'Task', 'Laser_Power_PT_0', 'Laser_Power_PT_1', 
                   'PC_Pressure', 'Ref_Cell_Flow', 'NO_Cell_Flow', 
                   'NO2_Cell_Flow', 'Sig_Cell_C_Flow', 'Inlet_ZA_MFC_Read', 
                   'Cal_ZA_MFC_Read', 'Ref_ZA_MFC_Read', 'Cal_NO_MFC_Read', 
                   'ZA_SB_MFC_Read', 'Ref_NO_MFC_Read', 'Cal_SB_MFC_Read', 
                   'Cal_NO_MFC_set', 'Inlet_ZA_MFC_set', 'BLC_0_flag', 
                   'BLC_1_flag', 'GPT_enable', 'T_board_main_LD1', 
                   'T_board_main_LD2', 'T_board_LIF_cell', 'T_board_REF_cell', 
                   'T_board_LSRBOX_air', 'T_board_NLO'] 

processing_variables = pd.read_csv(os.path.join(data_dir, 'processing_variables.txt'))

HK_data_dict = lif.import_HK_data(data_dir, day_folders)

for i in processing_variables.index:
    
    lif.reprocess_binary_data(
        data_freq = 10 # can take 10 or 100
        , date = str(processing_variables['date'][i])
        , file = str(processing_variables['bin_filename'][i])
        , log_start_datetime = str(processing_variables['log_start_datetime'][i])
        , data_dir = data_dir
        , HK_data_dict = HK_data_dict
        , channel_count = channel_count
        , channel_format = channel_format 
        , HK_headers_dict = HK_headers_dict
    )

        
        
        
    
    