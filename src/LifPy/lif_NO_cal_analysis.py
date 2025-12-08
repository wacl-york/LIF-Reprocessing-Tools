import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ref_cts_diff_limit = 0.75000 # lower limit ref_diff_cts_norm
pre_taskswitch = 300   # data points before task switch to ignore
post_taskswitch = 600   # data points after task switch to ignore
pre_peakfind = 20   # data points before ref_cts_diff drop to ignore
post_peakfind = 200   # data points after ref_cts_diff drop to ignore
no_cylinder_conc = 5000

data_dir = ('C:/Users/pp835/OneDrive - University of York/Documents/'
       'Data Analysis/CARES/Mace Head Full Data Analysis/Data'
       )
day_folders = lif.find_day_folders(data_dir)
cts_data = lif.read_processed_files(data_dir, day_folders)


# ---------------- Ref correction for CARES due to saturation ----------------

epsilon = 1e-10  # A very small number close to zero

limit = 90549
log_arg = 1 - (cts_data['ref_diff_cts'] / limit)

# Clip the argument for the logarithm calculation 
# This forces all values slightly above zero, preventing a runtime warning.
log_arg_clipped = np.clip(log_arg, a_min=epsilon, a_max=None)

# Calculate the raw result using the clipped data
result_raw = (182735 * (-np.log(log_arg_clipped) / 2.02)) / 100000

cts_data['ref_diff_cts'] = np.where(
    log_arg > 0,
    result_raw,
    np.nan
)
cts_data['ref_diff_cts_norm'] = (cts_data['ref_diff_cts']/cts_data['lsr_pwr_mW'])

# ---------------------- End of CARES NO Ref correction ----------------------


cts_data_ref_norm = lif.ref_normalise(cts_data, channels=['sig_A', 'sig_B'])
cts_data_flagged = lif.set_flags(cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind, post_peakfind, ref_cts_diff_limit)
cts_data_zeroed = lif.zero_correct_average(cts_data_flagged, channels=['sig_A', 'sig_B'], plot=True)
cts_data_MRs = cts_data_zeroed.copy()

###############################################################################
#-----------------Tested and optimised up to this point -----------------------
###############################################################################


cts_data_MRs['NO_mr'] = (cts_data_MRs["Cal_NO_MFC_Read"] / (cts_data_MRs["NO_Cell_Flow"]+cts_data_MRs['NO2_Cell_Flow']) * no_cylinder_conc)

lif.analyse_cals_robust(cts_data_MRs, plot=True, max_conc=5000
                 , channels=['sig_A', 'sig_B'], data_dir=data_dir
                 , molecule='NO')

lif.analyse_BLC_cals(cts_data_MRs, data_dir, plot=False)



cal_A_df = pd.read_csv(os.path.join(data_dir, 'sig_A_cal_data.txt'))

cal_A_df['cal_start_date_time'] = pd.to_datetime(cal_A_df['cal_start_date_time'])
cal_A_df = cal_A_df[(cal_A_df['slope_ref_norm'] > 1)
                    & (cal_A_df['slope_ref_norm'] < 2.8)
                    ]

cal_B_df = pd.read_csv(os.path.join(data_dir, 'sig_B_cal_data.txt'))

cal_B_df['cal_start_date_time'] = pd.to_datetime(cal_B_df['cal_start_date_time'])
cal_B_df = cal_B_df[(cal_B_df['slope_ref_norm'] > 1)
                     & (cal_B_df['slope_ref_norm'] < 2.8)
                     ]

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

cell_A_cal_factor_1 = cal_A_df_1['slope_ref_norm'].mean()
cell_A_cal_factor_2 = cal_A_df_2['slope_ref_norm'].mean()
cell_B_cal_factor_1 = cal_B_df_1['slope_ref_norm'].mean()
cell_B_cal_factor_2 = cal_B_df_2['slope_ref_norm'].mean()


cts_data_MRs['sig_A_cal_factor'] = cell_A_cal_factor_1
cts_data_MRs['sig_A_cal_factor'] = np.where(
    cts_data_MRs['Date_time'] >= split_time
    , cell_A_cal_factor_2
    , cts_data_MRs['sig_A_cal_factor']
    )

cts_data_MRs['sig_B_cal_factor'] = cell_B_cal_factor_1
cts_data_MRs['sig_B_cal_factor'] = np.where(
    cts_data_MRs['Date_time'] >= split_time
    , cell_B_cal_factor_2
    , cts_data_MRs['sig_B_cal_factor']
    )

cts_data_MRs['amb_NO_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0)
    , cts_data_MRs['sig_A_diff_cts_ref_norm_zero_corr'] / cts_data_MRs['sig_A_cal_factor']
    , np.nan
    )

cts_data_MRs['amb_NO_ppt'] = np.where(
    ((cts_data_MRs['Date_time'] >= pd.to_datetime('10/06/2025 01:10:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('10/06/2025 01:45:00', format='%d/%m/%Y %H:%M:%S')))
    | ((cts_data_MRs['Date_time'] >= pd.to_datetime('16/06/2025 01:15:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('16/06/2025 01:50:00', format='%d/%m/%Y %H:%M:%S')))
    , np.nan
    , cts_data_MRs['amb_NO_ppt']
    )

cts_data_MRs['amb_NOx_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0)
    , cts_data_MRs['sig_B_diff_cts_ref_norm_zero_corr'] / cts_data_MRs['sig_B_cal_factor']
    , np.nan
    )

cts_data_MRs['amb_NOx_ppt'] = np.where(
    ((cts_data_MRs['Date_time'] >= pd.to_datetime('10/06/2025 01:10:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('10/06/2025 01:45:00', format='%d/%m/%Y %H:%M:%S')))
    | ((cts_data_MRs['Date_time'] >= pd.to_datetime('16/06/2025 01:15:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('16/06/2025 01:50:00', format='%d/%m/%Y %H:%M:%S')))
    , np.nan
    , cts_data_MRs['amb_NOx_ppt']
    )

cts_data_MRs['amb_NO2_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0) & (cts_data_MRs['Date_time'] > pd.to_datetime('2025/06/06 00:00'))
    , (cts_data_MRs['amb_NOx_ppt'] - cts_data_MRs['amb_NO_ppt']) / 0.805
    , np.nan
    )



cts_data_MRs.set_index('Date_time')
cts_data_MRs_1min = cts_data_MRs.resample('1min').mean()
cts_data_MRs_5min = cts_data_MRs.resample('5min').mean()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cts_data_MRs_1min['Date_time'],cts_data_MRs_1min['amb_NO_ppt'], label='NO')
ax.plot(cts_data_MRs_1min['Date_time'],cts_data_MRs_1min['amb_NO2_ppt'], label='NO2')
ax.set_xlabel('date_time')
ax.set_ylabel('concentration / ppt')
plt.legend()
plt.show()

NO_prelim_data = cts_data_MRs_1min[['Date_time', 'amb_NO_ppt', 'amb_NO2_ppt']].copy()
NO_prelim_data.to_csv('NOx_CARES_MaceHead_prelim.txt', mode='a', header=False, index=False, sep=',')


NOx_data = NO_prelim_data

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(pd.to_datetime(NOx_data['Date_time']),NOx_data['amb_NO_ppt'], label='NO')
ax.plot(pd.to_datetime(NOx_data['Date_time']),NOx_data['amb_NO2_ppt'], label='NO2')
ax.set_xlabel('date_time')
ax.set_ylabel('concentration / ppt')
plt.legend()
plt.show()





fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['avg_lsr_pwr'],cal_A_df['slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['avg_lsr_pwr'],cal_B_df['slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.set_xlabel('average laser power during cal period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['cal_start_date_time'],cal_A_df['slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
#ax.plot(cal_B_df['cal_start_date_time'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.hlines(y=cell_A_cal_factor_1, xmin=min_time, xmax=split_time)
ax.hlines(y=cell_A_cal_factor_2, xmin=split_time, xmax=max_time)
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(cal_B_df_2['slope_ref_norm'], bins=8)
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