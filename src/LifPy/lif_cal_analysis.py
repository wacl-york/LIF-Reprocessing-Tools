import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ref_cts_diff_limit = 0.75000 # lower limit ref_diff_cts_norm
pre_taskswitch = 300   # data points before task switch to ignore
post_taskswitch = 600   # data points after task switch to ignore
pre_peakfind = 20   # data points before ref_cts_diff drop to ignore
post_peakfind = 200   # data points after ref_cts_diff drop to ignore
no_cylinder_conc = 5000
path = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\data_processed\\'
       )

cts_data = lif.read_processed_files(path)

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

cts_data_flagged = lif.set_flags_vectorised(cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind, post_peakfind, ref_cts_diff_limit)

cts_data_zeroed = lif.zero_correct(cts_data_flagged, channels=['sig_A', 'sig_B'], plot=True)



Std_cal_summary_A, Refnorm_cal_summary_A = lif.analyse_cals(
    cts_data_zeroed, plot=False, max_conc=5000, cell='A', path=path
    , molecule='NO')

Std_cal_summary_B, Refnorm_cal_summary_B = lif.analyse_cals(
    cts_data_zeroed, plot=False, max_conc=5000, cell='B', path=path
    , molecule='NO')

lif.analyse_BLC_cals(cts_data_zeroed, plot=True)







cal_A_df = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
                       'data_processed\\cell_A_cal_data.txt'
       )

cal_A_df['cal_start_date_time'] = pd.to_datetime(cal_A_df['cal_start_date_time'])
cal_A_df = cal_A_df[(cal_A_df['cell_A_slope_ref_norm'] > 1)
                    & (cal_A_df['cell_A_slope_ref_norm'] < 2.8)
                    ]

cal_B_df = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
                       'data_processed\\cell_B_cal_data.txt'
       )

cal_B_df['cal_start_date_time'] = pd.to_datetime(cal_B_df['cal_start_date_time'])
cal_B_df = cal_B_df[(cal_B_df['cell_B_slope_ref_norm'] > 1)
                     & (cal_B_df['cell_B_slope_ref_norm'] < 2.8)
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

cell_A_cal_factor_1 = cal_A_df_1['cell_A_slope_ref_norm'].mean()
cell_A_cal_factor_2 = cal_A_df_2['cell_A_slope_ref_norm'].mean()
cell_B_cal_factor_1 = cal_B_df_1['cell_B_slope_ref_norm'].mean()
cell_B_cal_factor_2 = cal_B_df_2['cell_B_slope_ref_norm'].mean()


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

cts_data_MRs['amb_NOx_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0)
    , cts_data_MRs['sig_B_diff_cts_ref_norm_zero_corr'] / cts_data_MRs['cell_B_cal_factor']
    , np.nan
    )

cts_data_MRs['amb_NO2_ppt'] = np.where(
    (cts_data_MRs['Task'] == 0) & (cts_data_MRs['Peak_find_flag'] == 0)
    , (cts_data_MRs['amb_NOx_ppt'] - cts_data_MRs['amb_NO_ppt']) / 0.81
    , np.nan
    )

cts_data_MRs.set_index('Date_time')
cts_data_MRs_1min = cts_data_MRs.resample('1min').mean()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cts_data_MRs_1min['Date_time'],cts_data_MRs_1min['amb_NO_ppt'], label='NO')
ax.plot(cts_data_MRs_1min['Date_time'],cts_data_MRs_1min['amb_NO2_ppt'], label='NO2')
ax.set_xlabel('date_time')
ax.set_ylabel('concentration / ppt')
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['avg_lsr_pwr'],cal_A_df['cell_A_slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['avg_lsr_pwr'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.set_xlabel('average laser power during cal period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['cal_start_date_time'],cal_A_df['cell_A_slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
#ax.plot(cal_B_df['cal_start_date_time'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.hlines(y=cell_A_cal_factor_1, xmin=min_time, xmax=split_time)
ax.hlines(y=cell_A_cal_factor_2, xmin=split_time, xmax=max_time)
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(cal_B_df_2['cell_B_slope_ref_norm'], bins=8)
plt.title('cell B period 2')
plt.show()


BLC_df = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
                       'data_processed\\BLC_cal_data.txt'
       )
BLC_df['cal_start_date_time'] = pd.to_datetime(BLC_df['cal_start_date_time'])

BLC_df = BLC_df[(BLC_df['conversion_efficiency'] > 0) & (BLC_df['BLC_V'] == 1.0)]# & (BLC_df['cal_start_date_time'] > pd.to_datetime('2025/06/14 12:00'))]

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