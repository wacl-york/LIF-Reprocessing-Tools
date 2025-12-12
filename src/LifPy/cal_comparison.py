import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = ('C:/Users/pp835/OneDrive - University of York/Documents/'
       'Data Analysis/CARES/Mace Head Full Data Analysis/Data'
       )

old_cals_A = pd.read_csv(os.path.join(data_dir, 'hide', 'sig_A_cal_data_old_function.txt'))
old_cals_A = (old_cals_A[old_cals_A['R2']>=0.75]).sort_values(by='cal_start_date_time')
old_cals_B = pd.read_csv(os.path.join(data_dir, 'hide', 'sig_B_cal_data_old_function.txt'))
old_cals_B = (old_cals_A[old_cals_B['R2']>=0.75]).sort_values(by='cal_start_date_time')
new_cals_A = pd.read_csv(os.path.join(data_dir, 'sig_A_cal_data.txt'))
new_cals_A = (new_cals_A[new_cals_A['R2']>=0.75]).sort_values(by='cal_start_date_time')
new_cals_B = pd.read_csv(os.path.join(data_dir, 'sig_B_cal_data.txt'))
new_cals_B = (new_cals_B[new_cals_B['R2']>=0.75]).sort_values(by='cal_start_date_time')

fig, ax = plt.subplots(2, 1, figsize=(12,12))

ax[0].plot(pd.to_datetime(old_cals_A['cal_start_date_time']), old_cals_A['slope_ref_norm'], color='orange', label='old cals')
ax[0].plot(pd.to_datetime(new_cals_A['cal_start_date_time']), new_cals_A['slope'], color='blue', label='new cals')
ax[0].fill_between(new_cals_A['cal_start_date_time'],
                   new_cals_A['slope'] - new_cals_A['slope_std_err'], # Lower Bound
                   new_cals_A['slope'] + new_cals_A['slope_std_err'], # Upper Bound
                   color='blue', alpha=0.2, label='_nolegend_')
ax[0].set_xlabel('cal start time')
ax[0].set_ylabel('calibration factor')
ax[0].set_title('cell A')
ax[0].legend()

ax[1].plot(pd.to_datetime(old_cals_B['cal_start_date_time']), old_cals_B['slope_ref_norm'], color='orange', label='old cals')
ax[1].plot(pd.to_datetime(new_cals_B['cal_start_date_time']), new_cals_B['slope'], color='blue', label='new cals')
ax[1].fill_between(new_cals_B['cal_start_date_time'],
                   new_cals_B['slope'] - new_cals_B['slope_std_err'], # Lower Bound
                   new_cals_B['slope'] + new_cals_B['slope_std_err'], # Upper Bound
                   color='blue', alpha=0.2, label='_nolegend_')
ax[1].set_xlabel('cal start time')
ax[1].set_ylabel('calibration factor')
ax[1].set_title('cell B')
ax[1].legend()

plt.tight_layout()
plt.show()





BLC_cals_flow = pd.read_csv(os.path.join(data_dir, 'BLC_cal_data_flow.txt'))
BLC_cals_reg = pd.read_csv(os.path.join(data_dir, 'BLC_cal_data_reg.txt'))
BLC_cals_old_code = pd.read_csv(os.path.join(data_dir, 'hide', 'BLC_cal_data_old_function.txt'))

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(pd.to_datetime(BLC_cals_flow['cal_start_date_time']), BLC_cals_flow['conversion_efficiency'], color='orange', label='flow corrected')
ax.plot(pd.to_datetime(BLC_cals_reg['cal_start_date_time']), BLC_cals_reg['conversion_efficiency'], color='blue', label='not flow corrected')
ax.plot(pd.to_datetime(BLC_cals_old_code['cal_start_date_time']), BLC_cals_old_code['conversion_efficiency'], color='green', label='old code')
ax.set_xlabel('cal start time')
ax.set_ylabel('conversion efficiency')
ax.set_title('BLC conversion efficiencies')
ax.legend()

plt.tight_layout()
plt.show()