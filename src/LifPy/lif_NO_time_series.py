import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- Spike Removal Function ---
def remove_spikes(df, column, percentile=99.5):
    """
    Replaces values above a given percentile threshold with NaN.
    Used to remove non-physical spikes.
    """
    # Calculate the threshold for the specified column, ignoring NaNs
    threshold = df[column].quantile(percentile / 100)
    # Replace values above the threshold with NaN
    df[column] = np.where(df[column] > threshold, np.nan, df[column])
    print(f"Spike removal applied to {column}: Threshold set at {threshold:.2f} ppt.")
    return df

ref_cts_diff_limit = 0.75000 # lower limit ref_diff_cts_norm
pre_taskswitch = 300    # data points before task switch to ignore
post_taskswitch = 600    # data points after task switch to ignore
pre_peakfind = 20    # data points before ref_cts_diff drop to ignore
post_peakfind = 200    # data points after ref_cts_diff drop to ignore
no_cylinder_conc = 5000
data_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
        'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\Data'
        )
day_folders = lif.find_day_folders(data_dir)

cts_data = lif.read_processed_files(data_dir, day_folders)

# Ref correction for CARES due to saturation
limit = 90549
log_arg = 1 - (cts_data['ref_diff_cts'] / limit)

cts_data['ref_diff_cts'] = np.where(
    log_arg > 0,
    (182735 * (-np.log(log_arg) / 2.02)) / 100000,
    np.nan
)
    
cts_data['ref_diff_cts_norm'] = (cts_data['ref_diff_cts']/cts_data['lsr_pwr_mW'])
cts_data['NO_mr'] = (cts_data["Cal_NO_MFC_Read"] / (cts_data["NO_Cell_Flow"]+cts_data['NO2_Cell_Flow']) * no_cylinder_conc)

cts_data_ref_norm = lif.ref_normalise(cts_data, channels=['sig_A', 'sig_B'])

cts_data_flagged = lif.set_flags(cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind, post_peakfind, ref_cts_diff_limit)

cts_data_zeroed = lif.zero_correct_average(cts_data_flagged, channels=['sig_A', 'sig_B'], plot=True)


# =============================================================================
# Std_cal_summary_A, Refnorm_cal_summary_A = lif.analyse_cals(
#     cts_data_zeroed, plot=False, max_conc=5000, cell='A', path=data_dir
#     , molecule='NO')
# 
# Std_cal_summary_B, Refnorm_cal_summary_B = lif.analyse_cals(
#     cts_data_zeroed, plot=False, max_conc=5000, cell='B', path=data_dir
#     , molecule='NO')
# =============================================================================

#lif.analyse_BLC_cals(cts_data_zeroed, data_dir, plot=False)







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


cts_data_MRs = cts_data_zeroed.copy()

cts_data_MRs['cell_A_cal_factor'] = cell_A_cal_factor_1
cts_data_MRs['cell_A_cal_factor'] = np.where(
    cts_data_MRs['Date_time'] >= split_time
    , cell_A_cal_factor_2
    , cts_data_MRs['cell_A_cal_factor']
    )

cts_data_MRs['cell_B_cal_factor'] = cell_B_cal_factor_1
cts_data_MRs['cell_B_cal_factor'] = np.where(
    cts_data_MRs['Date_time'] >= split_time
    , cell_B_cal_factor_2
    , cts_data_MRs['cell_B_cal_factor']
    )

cts_data_MRs['amb_NO_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0)
    , cts_data_MRs['sig_A_diff_cts_ref_norm_zero_corr'] / cts_data_MRs['cell_A_cal_factor']
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
    , cts_data_MRs['sig_B_diff_cts_ref_norm_zero_corr'] / cts_data_MRs['cell_B_cal_factor']
    , np.nan
    )

cts_data_MRs['amb_NOx_ppt'] = np.where(
    ((cts_data_MRs['Date_time'] >= pd.to_datetime('10/06/2025 01:10:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('10/06/2025 01:45:00', format='%d/%m/%Y %H:%M:%S')))
    | ((cts_data_MRs['Date_time'] >= pd.to_datetime('16/06/2025 01:15:00', format='%d/%m/%Y %H:%M:%S')) & (cts_data_MRs['Date_time'] <= pd.to_datetime('16/06/2025 01:50:00', format='%d/%m/%Y %H:%M:%S')))
    , np.nan
    , cts_data_MRs['amb_NOx_ppt']
)

cts_data_MRs['amb_NO2_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0) & (cts_data_MRs['Date_time'] > pd.to_datetime(
        '2025/06/06 00:00')), (cts_data_MRs['amb_NOx_ppt'] - cts_data_MRs['amb_NO_ppt']) / 0.805, np.nan
)

# --- Apply Spike Removal (Positive outliers) ---
# Note: Spike removal is applied to the raw data before averaging
cts_data_MRs = remove_spikes(cts_data_MRs, 'amb_NO_ppt')
cts_data_MRs = remove_spikes(cts_data_MRs, 'amb_NO2_ppt')

# --- Remove large negative values (as requested) ---
# Set a hard floor to remove clearly non-physical large negative outliers.
# Values below -100 ppt are set to NaN. Small negative values are retained.
negative_threshold = -100
cts_data_MRs['amb_NO_ppt'] = np.where(
    cts_data_MRs['amb_NO_ppt'] < negative_threshold,
    np.nan,
    cts_data_MRs['amb_NO_ppt']
)
cts_data_MRs['amb_NO2_ppt'] = np.where(
    cts_data_MRs['amb_NO2_ppt'] < negative_threshold,
    np.nan,
    cts_data_MRs['amb_NO2_ppt']
)


# --- Time Averaging (Resampling) ---
# Set 'Date_time' as the index for resampling to work correctly
cts_data_MRs = cts_data_MRs.set_index('Date_time')

# Perform resampling and reset index for plotting/export
cts_data_MRs_1min = cts_data_MRs.resample('1min').mean().reset_index()
cts_data_MRs_5min = cts_data_MRs.resample('5min').mean().reset_index()
cts_data_MRs_30min = cts_data_MRs.resample('30min').mean().reset_index()


# --- Plot: Time Series (Stacked Subplots, using 30-min average with consistent colors) ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Ambient NO (Consistent Color: indigo)
ax[0].plot(cts_data_MRs_30min['Date_time'], cts_data_MRs_30min['amb_NO_ppt'],
           label='NO', color='#4f46e5', linewidth=1.5)
ax[0].set_title('Time Series of Ambient NO and NO$_2$ (30-min average)',
                fontsize=14, fontweight='bold')
ax[0].set_ylabel('NO Concentration (ppt)', fontsize=12)
ax[0].legend(frameon=True, shadow=True, fancybox=True)
# ax[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Ambient NO2 (Consistent Color: red)
ax[1].plot(cts_data_MRs_30min['Date_time'], cts_data_MRs_30min['amb_NO2_ppt'],
           label='NO$_2$', color='#dc2626', linewidth=1.5)
ax[1].set_xlabel('Date Time', fontsize=12)
ax[1].set_ylabel('NO$_2$ Concentration (ppt)', fontsize=12)
ax[1].legend(frameon=True, shadow=True, fancybox=True)
# ax[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- Exporting 1-min data (unchanged) ---
NO_prelim_data = cts_data_MRs_1min[[
    'Date_time', 'amb_NO_ppt', 'amb_NO2_ppt']].copy()
NO_prelim_data.to_csv('NOx_CARES_MaceHead_prelim.txt',
                      mode='a', header=False, index=False, sep=',')


# --- Remaining plots (unchanged) ---

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cal_A_df['avg_lsr_pwr'], cal_A_df['slope_ref_norm'],
        marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['avg_lsr_pwr'], cal_B_df['slope_ref_norm'],
        marker='o', linestyle=' ', label='cell B')
ax.set_xlabel('average laser power during cal period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cal_A_df['cal_start_date_time'], cal_A_df['slope_ref_norm'],
        marker='o', linestyle=' ', label='cell A')
# ax.plot(cal_B_df['cal_start_date_time'],cal_B_df['slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.hlines(y=cell_A_cal_factor_1, xmin=min_time, xmax=split_time)
ax.hlines(y=cell_A_cal_factor_2, xmin=split_time, xmax=max_time)
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(cal_B_df_2['slope_ref_norm'], bins=8)
plt.title('cell B period 2')
plt.show()


BLC_df = pd.read_csv(os.path.join(data_dir, 'BLC_cal_data.txt'))
BLC_df['cal_start_date_time'] = pd.to_datetime(BLC_df['cal_start_date_time'])

BLC_df = BLC_df[(BLC_df['conversion_efficiency'] > 0) & (BLC_df['BLC_V'] == 1.0) & (
    BLC_df['cal_start_date_time'] > pd.to_datetime('2025/06/06 00:00'))]

conv_eff_avg = BLC_df['conversion_efficiency'].mean()
print(conv_eff_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(BLC_df['cal_start_date_time'], BLC_df['conversion_efficiency'],
        marker='o', linestyle=' ', label='BLC')
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('conversion efficiency')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(BLC_df['conversion_efficiency'], bins=8)
plt.title('BLC')
plt.show()
