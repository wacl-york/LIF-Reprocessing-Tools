import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

molecule = 'NO'
channels = {'sig_A':'NO', 'sig_B':'NO2'} # must be a dict of all sig channels
cal_task = 5
BLC = True
plot = False
averaging = '1_min'
filename = ' '
cal_cylinder_conc = 5000
pre_taskswitch = 300   # data points before task switch to ignore
post_taskswitch = 600   # data points after task switch to ignore
pre_peakfind = 20   # data points before ref_cts_diff drop to ignore
post_peakfind = 200   # data points after ref_cts_diff drop to ignore
ref_cts_diff_limit = 75000 # lower limit ref_diff_cts_norm

data_dir = ('C:/Users/pp835/OneDrive - University of York/Documents/'
            'Data Analysis/CARES/Mace Head Full Data Analysis/Data'
            )



day_folders = lif.find_day_folders(
    data_dir
    )
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )
cts_data = lif.CARES_NO_ref_correction(
    cts_data
    )
cts_data_ref_norm = lif.ref_normalise(
    cts_data, channels=channels
    )
cts_data_flagged = lif.set_flags(
    cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind
    , post_peakfind, ref_cts_diff_limit
    )
cts_data_zeroed = lif.zero_correct_average(
    cts_data_flagged, channels=channels, plot=plot
    )
# lif.analyse_cals(
#     cts_data_zeroed, data_dir, channels=['sig_A', 'sig_B'], molecule=molecule
#     , cal_cylinder_conc=cal_cylinder_conc, plot=plot, save_csv=True, cal_task=5
#     )
lif.analyse_BLC_cals(
    
    )
# ------------------ Doesn't include NO2 conversion yet! ---------------------
MR_data = lif.apply_cals_average(
    data_dir, cts_data_zeroed, channels=channels, BLC=BLC
    )
# ----------------------------------------------------------------------------

MR_data_resampled = lif.resample_data(
    MR_data, averaging
    )
lif.save_to_csv(
    data_dir, filename, MR_data_resampled
    )

###############################################################################
#-----------------Tested and optimised up to this point -----------------------
###############################################################################

MR_data = cts_data_zeroed.copy()


#lif.analyse_BLC_cals(MR_data, data_dir, plot=False)



cal_A_df = pd.read_csv(os.path.join(data_dir, 'sig_A_cal_data.txt'))

cal_A_df['cal_start_date_time'] = pd.to_datetime(cal_A_df['cal_start_date_time'])
# =============================================================================
# cal_A_df = cal_A_df[(cal_A_df['slope'] > 1)
#                     & (cal_A_df['slope'] < 2.8)
#                     ]
# =============================================================================

cal_B_df = pd.read_csv(os.path.join(data_dir, 'sig_B_cal_data.txt'))

cal_B_df['cal_start_date_time'] = pd.to_datetime(cal_B_df['cal_start_date_time'])
# =============================================================================
# cal_B_df = cal_B_df[(cal_B_df['slope'] > 1)
#                      & (cal_B_df['slope'] < 2.8)
#                      ]
# =============================================================================

split_time = pd.to_datetime('2025/06/14 12:00:00')
min_time = pd.to_datetime(cal_A_df['cal_start_date_time'].min())
max_time = pd.to_datetime(cal_A_df['cal_start_date_time'].max())

cal_A_mask_1 = (cal_A_df['cal_start_date_time'] < split_time)
cal_A_mask_2 = (cal_A_df['cal_start_date_time'] > split_time)
cal_B_mask_1 = (cal_B_df['cal_start_date_time'] < split_time)
cal_B_mask_2 = (cal_B_df['cal_start_date_time'] > split_time)

cal_A_df_1 = cal_A_df[cal_A_mask_1]
cal_A_df_2 = cal_A_df[cal_A_mask_2]
cal_B_df_1 = cal_B_df[cal_B_mask_1]
cal_B_df_2 = cal_B_df[cal_B_mask_2]

cell_A_cal_factor_1 = cal_A_df_1['slope'].mean()
cell_A_cal_factor_2 = cal_A_df_2['slope'].mean()
cell_B_cal_factor_1 = cal_B_df_1['slope'].mean()
cell_B_cal_factor_2 = cal_B_df_2['slope'].mean()


MR_data['sig_A_cal_factor'] = cell_A_cal_factor_1
MR_data['sig_A_cal_factor'] = np.where(
    MR_data['Date_time'] >= split_time
    , cell_A_cal_factor_2
    , MR_data['sig_A_cal_factor']
    )

MR_data['sig_B_cal_factor'] = cell_B_cal_factor_1
MR_data['sig_B_cal_factor'] = np.where(
    MR_data['Date_time'] >= split_time
    , cell_B_cal_factor_2
    , MR_data['sig_B_cal_factor']
    )

MR_data['amb_NO_ppt'] = np.where(
    (MR_data['Task'] == 0) & (MR_data['Peak_find_flag'] == 0)
    , MR_data['sig_A_diff_cts_ref_norm_zero_corr'] / MR_data['sig_A_cal_factor']
    , np.nan
    )

MR_data['amb_NO_ppt'] = np.where(
    ((MR_data['Date_time'] >= pd.to_datetime('10/06/2025 01:10:00', format='%d/%m/%Y %H:%M:%S')) & (MR_data['Date_time'] <= pd.to_datetime('10/06/2025 01:45:00', format='%d/%m/%Y %H:%M:%S')))
    | ((MR_data['Date_time'] >= pd.to_datetime('16/06/2025 01:15:00', format='%d/%m/%Y %H:%M:%S')) & (MR_data['Date_time'] <= pd.to_datetime('16/06/2025 01:50:00', format='%d/%m/%Y %H:%M:%S')))
    , np.nan
    , MR_data['amb_NO_ppt']
    )

MR_data['amb_NOx_ppt'] = np.where(
    (MR_data['Task'] == 0) & (MR_data['Peak_find_flag'] == 0)
    , MR_data['sig_B_diff_cts_ref_norm_zero_corr'] / MR_data['sig_B_cal_factor']
    , np.nan
    )

MR_data['amb_NOx_ppt'] = np.where(
    ((MR_data['Date_time'] >= pd.to_datetime('10/06/2025 01:10:00', format='%d/%m/%Y %H:%M:%S')) & (MR_data['Date_time'] <= pd.to_datetime('10/06/2025 01:45:00', format='%d/%m/%Y %H:%M:%S')))
    | ((MR_data['Date_time'] >= pd.to_datetime('16/06/2025 01:15:00', format='%d/%m/%Y %H:%M:%S')) & (MR_data['Date_time'] <= pd.to_datetime('16/06/2025 01:50:00', format='%d/%m/%Y %H:%M:%S')))
    , np.nan
    , MR_data['amb_NOx_ppt']
    )

MR_data['amb_NO2_ppt'] = np.where(
    (MR_data['Task'] == 0) & (MR_data['Peak_find_flag'] == 0) & (MR_data['Date_time'] > pd.to_datetime('2025/06/06 00:00'))
    , (MR_data['amb_NOx_ppt'] - MR_data['amb_NO_ppt']) / 0.805
    , np.nan
    )



MR_data.set_index('Date_time')
MR_data_1min = MR_data.resample('1min').mean()
MR_data_5min = MR_data.resample('5min').mean()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(MR_data_1min['Date_time'],MR_data_1min['amb_NO_ppt'], label='NO')
ax.plot(MR_data_1min['Date_time'],MR_data_1min['amb_NO2_ppt'], label='NO2')
ax.set_xlabel('date_time')
ax.set_ylabel('concentration / ppt')
plt.legend()
plt.show()

NO_prelim_data = MR_data_1min[['Date_time', 'amb_NO_ppt', 'amb_NO2_ppt']].copy()
NO_prelim_data.to_csv('NOx_CARES_MaceHead_prelim.txt', mode='a', header=True, index=False, sep=',')


NOx_data = NO_prelim_data

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(pd.to_datetime(NOx_data['Date_time']),NOx_data['amb_NO_ppt'], label='NO')
ax.plot(pd.to_datetime(NOx_data['Date_time']),NOx_data['amb_NO2_ppt'], label='NO2')
ax.set_xlabel('date_time')
ax.set_ylabel('concentration / ppt')
plt.legend()
plt.show()





fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['avg_lsr_pwr'],cal_A_df['slope'], marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['avg_lsr_pwr'],cal_B_df['slope'], marker='o', linestyle=' ', label='cell B')
ax.set_xlabel('average laser power during cal period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['cal_start_date_time'],cal_A_df['slope'], marker='o', linestyle=' ', label='cell A')
#ax.plot(cal_B_df['cal_start_date_time'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.hlines(y=cell_A_cal_factor_1, xmin=min_time, xmax=split_time)
ax.hlines(y=cell_A_cal_factor_2, xmin=split_time, xmax=max_time)
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(cal_B_df_2['slope'], bins=8)
plt.title('cell B period 2')
plt.show()


BLC_df = pd.read_csv(os.path.join(data_dir, 'BLC_cal_data.txt'))
BLC_df['cal_start_date_time'] = pd.to_datetime(BLC_df['cal_start_date_time'])

BLC_df = BLC_df[(BLC_df['conversion_efficiency'] > 0) & (BLC_df['BLC_V'] == 1.0) & (BLC_df['cal_start_date_time'] > pd.to_datetime('2025/06/06 00:00'))]

conv_eff_avg = BLC_df['conversion_efficiency'].mean()
print(conv_eff_avg)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(BLC_df['cal_start_date_time'],BLC_df['conversion_efficiency'], marker='o', linestyle=' ', label='BLC')
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('conversion efficiency')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(BLC_df['conversion_efficiency'], bins=8)
plt.title('BLC')
plt.show()