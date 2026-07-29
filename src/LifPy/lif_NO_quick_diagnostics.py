import lif_functions as lif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------- Variables to manually input --------------------------

molecule = 'NO'                                                                 # NO (even for multi-channel NOx) or SO2
channels = {'sig_A':'NO', 'sig_B':'NO2', 'sig_C':'NOy'}                         # dict of all sig channels
start_times = {'sig_A':'30/05/2025 00:00', 'sig_B':'05/06/2025 00:00'}
cal_task = 5                                                                    # Task number associated with regular cals
BLC_cal_task = 1                                                                # Task number associated with BLC cals
R2_limit = 0.75                                                                 # Lower limit for R2 values of cal regression
BLC = True                                                                      # Boolean indicator of whether BLC present
plot = True                                                                     # diagnostics plots at various analysis stages
averaging = '1 min'                                                             # averaging for the final output file
filename = 'LOKI_NOx_test_20260502_20260504'                                    # filename for the resampled MR output
cal_cylinder_conc = 5000                                                        # in ppb
pre_taskswitch = 300    # (at 10Hz = 30 secs)                                   # data points before task switch to ignore
post_taskswitch = 600   # (at 10Hz = 60 secs)                                   # data points after task switch to ignore
pre_peakfind = 20       # (at 10Hz = 2 secs)                                    # data points before ref_cts_diff drop to ignore
post_peakfind = 200     # (at 10Hz = 20 secs)                                   # data points after ref_cts_diff drop to ignore
ref_cts_diff_limit = 2000                                                       # lower limit ref_diff_cts_norm

data_dir = (r'C:\Users\pp835\OneDrive - University of York\Documents\Data Analysis\COCO-VOC\data')
# ----------------------------------------------------------------------------

day_folders = lif.find_day_folders(
    data_dir
    )
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )
    
cts_data['sig_A_diff_cts_ref_norm'] = cts_data['sig_A_diff_cts']/cts_data['ref_diff_cts']
cts_data['sig_B_diff_cts_ref_norm'] = cts_data['sig_B_diff_cts']/cts_data['ref_diff_cts']
cts_data['sig_C_diff_cts_ref_norm'] = cts_data['sig_C_diff_cts']/cts_data['ref_diff_cts']




fig_counts, ax_counts = plt.subplots(4, 1, figsize=(8, 20), sharex=True)

# Cell A Plot
ax_counts[0].plot(cts_data['Date_time'], cts_data['sig_A_diff_cts_norm'])
ax_counts[0].set_ylabel('sig A diff counts (laser norm)', fontsize=11)

# Cell B plot
ax_counts[1].plot(cts_data['Date_time'], cts_data['sig_B_diff_cts_norm'])
ax_counts[1].set_ylabel('sig B diff counts (laser norm)', fontsize=11)

# Cell C plot
ax_counts[2].plot(cts_data['Date_time'], cts_data['sig_C_diff_cts_norm'])
ax_counts[2].set_ylabel('sig C diff counts (laser norm)', fontsize=11)

# Ref Cell plot
ax_counts[3].plot(cts_data['Date_time'], cts_data['ref_diff_cts'], label='ref')
ax_counts[3].set_ylabel('ref counts (laser norm)', fontsize=11)

plt.tight_layout() 



fig_norm_counts, ax_norm_counts = plt.subplots(3, 1, figsize=(8, 15), sharex=True)

# Cell A Plot
ax_norm_counts[0].plot(cts_data['Date_time'], cts_data['sig_A_diff_cts_ref_norm'])
ax_norm_counts[0].set_ylabel('sig A diff counts ref norm (laser norm)', fontsize=11)

# Cell B plot
ax_norm_counts[1].plot(cts_data['Date_time'], cts_data['sig_B_diff_cts_ref_norm'])
ax_norm_counts[1].set_ylabel('sig B diff counts ref norm (laser norm)', fontsize=11)

# Cell C plot
ax_norm_counts[2].plot(cts_data['Date_time'], cts_data['sig_C_diff_cts_ref_norm'])
ax_norm_counts[2].set_ylabel('sig C diff counts ref norm (laser norm)', fontsize=11)

plt.tight_layout()



fig_laser, ax_laser = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

# laser power plot
ax_laser.plot(cts_data['Date_time'], cts_data['lsr_pwr_mW'])
ax_laser.set_ylabel('laser power', fontsize=11)

plt.tight_layout()



fig_flow, ax_flow = plt.subplots(3, 1, figsize=(8, 15), sharex=True)

# Cell A Plot
ax_flow[0].plot(cts_data['Date_time'], cts_data['NO_Cell_Flow'])
ax_flow[0].set_ylabel('cell A flow', fontsize=11)

# Cell B plot
ax_flow[1].plot(cts_data['Date_time'], cts_data['NO2_Cell_Flow'])
ax_flow[1].set_ylabel('cell B flow', fontsize=11)

# Cell C plot
ax_flow[2].plot(cts_data['Date_time'], cts_data['Sig_C_Cell_Flow'])
ax_flow[2].set_ylabel('cell C flow', fontsize=11)

plt.tight_layout()



fig_mfc, ax_mfc = plt.subplots(3, 1, figsize=(8, 15), sharex=True)

# cal NO
ax_mfc[0].plot(cts_data['Date_time'], cts_data['Cal_NO_MFC_Read'])
ax_mfc[0].set_ylabel('cal NO MFC', fontsize=11)

# Cal SB
ax_mfc[1].plot(cts_data['Date_time'], cts_data['Cal_SB_MFC_Read'])
ax_mfc[1].set_ylabel('cal SB MFC', fontsize=11)

plt.tight_layout()



fig_blc, ax_blc = plt.subplots(3, 1, figsize=(8, 15), sharex=True)

# BLC 0 LEDs
ax_blc[0].plot(cts_data['Date_time'], cts_data['BLC_0_flag'])
ax_blc[0].set_ylabel('BLC 0', fontsize=11)

# BLC 1 LEDs
ax_blc[1].plot(cts_data['Date_time'], cts_data['BLC_1_flag'])
ax_blc[1].set_ylabel('BLC 1', fontsize=11)

# GPT flag
ax_blc[2].plot(cts_data['Date_time'], cts_data['GPT_enable'])
ax_blc[2].set_ylabel('GPT', fontsize=11)

plt.tight_layout()



fig_task, ax_task = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

# laser power plot
ax_task.plot(cts_data['Date_time'], cts_data['Task'])
ax_task.set_ylabel('task', fontsize=11)

plt.tight_layout()



fig_comp, ax_comp = plt.subplots(3, 1, figsize=(8, 25), sharex=True)

# Cell A Plot
ax_comp[0].plot(cts_data['Date_time'], cts_data['sig_A_diff_cts_ref_norm'])
ax_comp[0].set_ylabel('sig A diff counts ref norm (laser norm)', fontsize=11)

# Cell C plot
ax_comp[1].plot(cts_data['Date_time'], cts_data['sig_C_diff_cts_ref_norm'])
ax_comp[1].set_ylabel('sig C diff counts ref norm (laser norm)', fontsize=11)

# cal NO
ax_comp[2].plot(cts_data['Date_time'], cts_data['Cal_NO_MFC_Read'])
ax_comp[2].set_ylabel('cal NO MFC', fontsize=11)

plt.tight_layout()
