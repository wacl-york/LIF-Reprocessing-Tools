import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ref_cts_diff_limit = 75000 # lower limit ref_diff_cts_norm
pre_taskswitch = 300   # data points before task switch to ignore
post_taskswitch = 600   # data points after task switch to ignore
pre_peakfind = 20   # data points before ref_cts_diff drop to ignore
post_peakfind = 200   # data points after ref_cts_diff drop to ignore
no_cylinder_conc = 5000
path = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\data_processed\\'
       )

cts_data = lif.read_processed_files(path)

cts_data['ref_diff_cts'] = (
    182735 * (-np.log(1-(cts_data['ref_diff_cts']/90549))/2.02)
    )
cts_data['ref_diff_cts_norm'] = (
    cts_data['ref_diff_cts']/cts_data['lsr_pwr_mW']
    )
cts_data['sig_A_diff_cts_ref_norm'] = (
    cts_data['sig_A_diff_cts']/cts_data['ref_diff_cts']
    )
cts_data['sig_B_diff_cts_ref_norm'] = (
    cts_data['sig_B_diff_cts']/cts_data['ref_diff_cts']
    )
cts_data['NO_mr'] = (
    cts_data["Cal_NO_MFC_Read"] / (cts_data["NO_Cell_Flow"]+cts_data['NO2_Cell_Flow']) \
    * no_cylinder_conc
    )

cts_data = lif.set_flags(cts_data, pre_taskswitch, post_taskswitch
                         , pre_peakfind, post_peakfind, ref_cts_diff_limit)
lif.gen_cal_files(path)

Std_cal_summary_A, Refnorm_cal_summary_A = lif.Analyse_cals(
    cts_data, plot=True, max_conc=5000, cell='A'
    )

Std_cal_summary_B, Refnorm_cal_summary_B = lif.Analyse_cals(
    cts_data, plot=True, max_conc=5000, cell='B'
    )

cal_A_df = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
                       'post_processing\\cal_analysis\\cell_A_cal_data.txt'
       )
cal_A_df['cal_start_date_time'] = pd.to_datetime(cal_A_df['cal_start_date_time'])
cal_A_df = cal_A_df[(cal_A_df['cell_A_slope_ref_norm'] > 1e-5)
                    & (cal_A_df['cell_A_slope_ref_norm'] < 2.8e-5)
                    ]

cal_B_df = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                       'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
                       'post_processing\\cal_analysis\\cell_B_cal_data.txt'
       )
cal_B_df['cal_start_date_time'] = pd.to_datetime(cal_B_df['cal_start_date_time'])
cal_B_df = cal_B_df[(cal_B_df['cell_B_slope_ref_norm'] > 1e-5)
                     & (cal_B_df['cell_B_slope_ref_norm'] < 2.8e-5)
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

print(cell_A_cal_factor_1, cell_A_cal_factor_2, cell_B_cal_factor_1, cell_B_cal_factor_2)


fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cal_A_df['avg_lsr_pwr'],cal_A_df['cell_A_slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['avg_lsr_pwr'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.set_xlabel('average laser power during cal period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(cal_A_df['cal_start_date_time'],cal_A_df['cell_A_slope_ref_norm'], marker='o', linestyle=' ', label='cell A')
ax.plot(cal_B_df['cal_start_date_time'],cal_B_df['cell_B_slope_ref_norm'], marker='o', linestyle=' ', label='cell B')
ax.hlines(y=cell_B_cal_factor_1, xmin=min_time, xmax=split_time)
ax.hlines(y=cell_B_cal_factor_2, xmin=split_time, xmax=max_time)
ax.set_xlabel('start time of calibration period')
ax.set_ylabel('calibration factor')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.hist(cal_B_df_1['cell_B_slope_ref_norm'], bins=8)
plt.title('cell B period 1')
plt.show()

