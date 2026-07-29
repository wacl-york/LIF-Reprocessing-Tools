'''

lif_NO_generate_processing_variables.py

Run this script once to generate the processing_variables.txt file which is 
required for binary reprocessing.
Details of file structure and how to run this code are found in a README file 
in the LIF-Reprocessing-Tools github.

'''
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

lif.gen_processing_var(data_dir, day_folders, channel_format, channel_count)

# =============================================================================
# lif.misaligned_counts(data_dir, day_folders, channel_format, channel_count
#                   , cal_task=5, molecule='NO', plot=True)
# =============================================================================

