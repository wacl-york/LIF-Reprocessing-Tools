import lif_functions as lif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------- Variables to manually input --------------------------

molecule = 'NO'                                                                 # NO (even for multi-channel NOx) or SO2
channels = {'sig_A':'NO', 'sig_B':'NO2'}                                        # dict of all sig channels
start_times = {'sig_A':'25/06/2026 00:00', 'sig_B':'25/06/2026 00:00'}
cal_task = 5                                                                    # Task number associated with regular cals
BLC_cal_task = 1                                                                # Task number associated with BLC cals
R2_limit = 0.40                                                                 # Lower limit for R2 values of cal regression
BLC = False                                                                     # Boolean indicator of whether BLC present
plot = True                                                                     # diagnostics plots at various analysis stages
averaging = '1 min'                                                             # averaging for the final output file
filename = 'LOKI_NOx_cocovoc'                                                   # filename for the resampled MR output
cal_cylinder_conc = 5000                                                        # in ppb
pre_taskswitch = 0    # (at 10Hz = 30 secs)                                     # data points before task switch to ignore
post_taskswitch = 0   # (at 10Hz = 60 secs)                                     # data points after task switch to ignore
pre_peakfind = 20       # (at 10Hz = 2 secs)                                    # data points before ref_cts_diff drop to ignore
post_peakfind = 200     # (at 10Hz = 20 secs)                                   # data points after ref_cts_diff drop to ignore
ref_cts_diff_limit = 2000                                                       # lower limit ref_diff_cts_norm

data_dir = (r'C:\Users\pp835\OneDrive - University of York\Documents\Data Analysis\COCO-VOC\NOx_data\Analysis')
# ----------------------------------------------------------------------------

day_folders = lif.find_day_folders(
    data_dir
    )
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )
cts_data_ref_norm = lif.ref_normalise(
    cts_data, channels=channels
    )
cts_data_flagged = lif.set_flags(
    cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind
    , post_peakfind, ref_cts_diff_limit
    )

# =============================================================================
# cts_data_usable = cts_data_flagged[cts_data_flagged['Peak_find_flag']==0]
# 
# fig, ax = plt.subplots(3, 1, figsize=(8, 15), sharex=True)
# 
# # Cell A Plot
# ax[0].plot(cts_data_usable['Date_time'], cts_data_usable['sig_A_diff_cts_ref_norm'])
# ax[0].set_ylabel('sig A diff counts (ref norm)', fontsize=11)
# 
# # Cell B plot
# ax[1].plot(cts_data_usable['Date_time'], cts_data_usable['sig_B_diff_cts_ref_norm'])
# ax[1].set_ylabel('sig B diff counts (ref norm)', fontsize=11)
# 
# # Cell C plot
# ax[2].plot(cts_data_usable['Date_time'], cts_data_usable['sig_C_diff_cts_ref_norm'])
# ax[2].set_ylabel('sig C diff counts (ref norm)', fontsize=11)
# 
# plt.tight_layout() 
# =============================================================================

cts_data_zero_periods = cts_data_flagged[cts_data_flagged['Task'] == 4]
cts_data_cal_periods = cts_data_flagged[cts_data_flagged['Task'] == 5]


cts_data_zeroed = lif.zero_correct_average(
    cts_data_flagged, channels=channels, plot=True
    )

cts_data_cal_periods = cts_data_zeroed[cts_data_flagged['Task'] == 5]



# =============================================================================
# cal_data = cts_data_zeroed.copy()
# 
# # Find all flow columns and sum them to a total flow
# flows = [column for column in cal_data.columns if column !='Sig_C_Cell_Flow'
#          if 'Flow' in column]
# cal_data['total_flow'] = cal_data[flows].sum(axis=1)
# # Calculate the MR associated with the cal gas and flow
# cal_data[f'{molecule}_mr'] = (
#     cal_data[f'Cal_{molecule}_MFC_Read'] / (cal_data['total_flow']
#     ) * cal_cylinder_conc)
# 
# # Isolate the columns of data required for the cal analysis
# diff_cts_columns = [f'{channel}_diff_cts_ref_norm_zero_corr'
#                     for channel in channels]
# additional_columns = ['Date_time', 'Task', 'lsr_pwr_mW',
#                       f'{molecule}_mr', f'Cal_{molecule}_MFC_Read',
#                       'Cal_SB_MFC_Read', f'Cal_{molecule}_MFC_set']
# all_columns = diff_cts_columns + additional_columns
# # Copy the dataframe containing only the columns of interest
# cal_data = cal_data[all_columns].copy()
# 
# # Find start of calibration periods (Task 5 starts)
# cal_data['start_of_cal'] = np.where(
#     (cal_data['Task'] == cal_task) & (cal_data['Task'].shift(1) != cal_task)
#     , 1
#     , 0
#     )
# # Cumulatively sum the start_of_cal flags to number the calibrations
# cal_data['cal_number'] = cal_data['start_of_cal'].cumsum()
# # Mask for calibration periods with cal SB off
# cal_task_mask = ((cal_data['Task'] == cal_task) &
#                  (cal_data['Cal_SB_MFC_Read'] < 0.01))
# cal_data = cal_data[cal_task_mask]
# =============================================================================

lif.analyse_cals(
    cts_data_zeroed, data_dir, channels=channels, molecule=molecule
    , cal_cylinder_conc=cal_cylinder_conc, plot=plot, save_csv=True
    , cal_task=cal_task
    )

cts_data_zeroed = cts_data_zeroed[(cts_data_zeroed['Task']==0) & (cts_data_zeroed['Peak_find_flag']==0)]

lif.analyse_BLC_cals(
    cts_data_zeroed, data_dir, plot=plot, save_csv=True
    , BLC_cal_task=BLC_cal_task
    )

MR_data = lif.apply_cals_average(
    data_dir, cts_data_zeroed, R2_limit, channels=channels, BLC=BLC
     , start_times=start_times
     )

MR_data['amb_NO2_ppt'] = (MR_data['amb_NO2_ppt'] - MR_data['amb_NO_ppt']) * 0.8


fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Cell A Plot
ax[0].plot(MR_data['Date_time'], MR_data['amb_NO_ppt'])
ax[0].set_ylabel('NO / ppt', fontsize=11)

# Cell B plot
ax[1].plot(MR_data['Date_time'], MR_data['amb_NO2_ppt'])
ax[1].set_ylabel('NO2 / ppt', fontsize=11)


plt.tight_layout() 








































# =============================================================================
# print('\nCalculating zero correction')
# 
# # use a mask to select all of the zero data associated with task 4
# # set the index to Date_time for averaging later
# cts_data_zero = cts_data_flagged.copy()
# start_of_zero = (cts_data_zero['Task'] == 4) & (cts_data_zero['Task'].shift(1) != 4)
# cts_data_zero['zero_number'] = start_of_zero.cumsum()
# cts_data_zero = cts_data_zero[(cts_data_zero['Task']==4) & (cts_data_zero['Peak_find_flag']==0)]
# grouped_zeros = cts_data_zero.groupby('zero_number')
# 
# # =============================================================================
# # if 'sig_B' in channels:
# # 
# #     cts_data_zero['sig_B_diff_cts_ref_norm'] = np.where(
# #         (cts_data_zero['BLC_0_flag'] == 1.0) & (cts_data_zero['BLC_1_flag'] == 1.0)
# #         , cts_data_zero['sig_B_diff_cts_ref_norm']
# #         , np.nan
# #         )
# # =============================================================================
# 
# for channel in channels:
# 
#     column_name = f'{channel}_diff_cts_ref_norm'
#     
#     zero_stats = grouped_zeros.agg({
#         column_name: ['mean', 'std', 'count'],
#         'Date_time': 'mean'
#         }).dropna()
#     
#     # Flattening the MultiIndex columns
#     zero_stats.columns = ['_'.join(col).strip() for col in zero_stats.columns.values]
#     
#     # Renaming for clarity
#     zero_stats = zero_stats.rename(columns={
#         'Date_time_mean': 'zero_midpoint'
#     })
#     
#     mean_zero = zero_stats[f'{column_name}_mean'].mean()
#     std_zero = zero_stats[f'{column_name}_mean'].std()
#     lower_limit = mean_zero - 3*std_zero
#     upper_limit = mean_zero + 3*std_zero
#     spike_mask = ((zero_stats[f'{column_name}_mean'] > lower_limit)
#         & (zero_stats[f'{column_name}_mean'] < upper_limit))
#     
#     zero_stats = zero_stats[spike_mask]
#     
#     mean_zero_spikes_removed = zero_stats[f'{column_name}_mean'].mean()
#     print(f'\n{channel}:\nmean zero before spike removal = {mean_zero}'
#           f'\nmean zero after spike removal = {mean_zero_spikes_removed}')
#     
#     mean_correction = mean_zero_spikes_removed # The overall mean after spike removal
#     correction_values = np.full(len(cts_data_flagged), mean_correction)
#     
#     cts_data_flagged[f'{channel}_zero_offset'] = correction_values
#     cts_data_flagged[f'{channel}_diff_cts_ref_norm_zero_corr'] = cts_data_flagged[column_name] - correction_values
#     print(f'zero correction applied to {channel}')
#     
#     if plot:
#         
#         fig, ax = plt.subplots( 2, 1, figsize=(12, 8))
#         
#         ax[0].errorbar(zero_stats['zero_midpoint']
#                          , zero_stats[f'{column_name}_mean']
#                          , yerr=zero_stats[f'{column_name}_std']
#                          , linestyle=''
#                          , marker='o'
#                          , markersize=2
#                          , capsize=2
#                          , label='zero measurement means, +/- 1std'
#                          )
#         #ax.plot(cts_data_zero['Date_time'], cts_data_zero[column_name], label='raw zero data')
#         ax[0].plot(cts_data_flagged['Date_time'], cts_data_flagged[f'{channel}_zero_offset'], label='zero correction')
#         ax[0].set_xlabel('Date_time')
#         ax[0].set_ylabel(column_name)
#         ax[0].set_title(f'{channel} zero correction')
#         ax[0].legend()
#         
#         ax[1].hist(zero_stats[f'{column_name}_mean'], bins=50)
#         ax[1].set_xlabel(column_name)
#         
#         plt.show()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# lif.analyse_cals(
#     cts_data_zeroed, data_dir, channels=channels, molecule=molecule
#     , cal_cylinder_conc=cal_cylinder_conc, plot=plot, save_csv=True
#     , cal_task=cal_task
#     )
# lif.analyse_BLC_cals(
#     cts_data_zeroed, data_dir, plot=plot, save_csv=True
#     , BLC_cal_task=BLC_cal_task
#     )
# MR_data = lif.apply_cals_average(
#     data_dir, cts_data_zeroed, R2_limit, channels=channels, BLC=BLC
#     , start_times=start_times
#     )
# 
# # mask = ((MR_data['Date_time'] < pd.to_datetime('06/06/2025 00:00', dayfirst=True)) |
# #         (MR_data['Date_time'] > pd.to_datetime('06/06/2025 12:00', dayfirst=True))
# #         )
# # MR_data = MR_data[mask]
# 
# # MR_data_resampled = lif.resample_data(
# #     MR_data, averaging
# #     )
# # lif.save_to_csv(
# #     data_dir, filename, MR_data_resampled
# #     )
# #lif.CARES_NO_plot_data_old(data_dir, MR_data)
# 
# 
# # lif.CARES_NO_diurnal_raw_mean(
# #     data_dir, MR_data
# #     )
# # lif.CARES_NO_diurnal_raw_median(
# #     data_dir, MR_data
# #     )
# # lif.CARES_NO_diurnal_median_of_medians(
# #     data_dir, MR_data
# #     )
# # lif.CARES_NO_diurnal_mean_of_medians(
# #     data_dir, MR_data
# #     )
# # lif.CARES_NO_diurnal_median_of_means(
# #     data_dir, MR_data
# #     )
# # lif.CARES_NO_diurnal_mean_of_means(
# #     data_dir, MR_data
# #     )
# =============================================================================
