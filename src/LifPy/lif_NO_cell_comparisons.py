'''
This code can be used to compare periods of data like-for-like (eg. for assessing the impact of cell changes on dark counts)
This uses HK data only, not binary counts data.
Variables to compare should be added to the plot_vars list, in the exact format as they are in the HK file headings.
The two periods to test should be named and their times specified in the first section
'''


import allantools

import numpy as np
import pandas as pd
import lif_functions as lif   
import matplotlib.pyplot as plt

from datetime import datetime as dt 


# --- specify the test periods  ---
test_period_1 = 'long arms, baffles'
test_period_2 = 'short arms, baffles, cage-mounted irises'

test_period_1_start = '13/01/2026 10:00'
test_period_1_end = '13/01/2026 13:00'
test_period_2_start = '15/01/2026 13:35'
test_period_2_end = '15/01/2026 15:35'

plot_vars = ['Signal_Dk_Counts', 'dk_cts_mW-1', 'dk_cts_norm']
allan_dev = True


# --- set-up and read in data ---
data_dir = (r'C:\Users\pp835\OneDrive - University of York\Documents\Data Analysis\CARES\Post campaign testing\Data')

day_folders = lif.find_day_folders(data_dir)
start_day = dt.strptime(test_period_1_start, '%d/%m/%Y %H:%M').strftime('%Y%m%d')
end_day = dt.strptime(test_period_2_end, '%d/%m/%Y %H:%M').strftime('%Y%m%d')
day_folders = [day for day in day_folders if start_day <= day <= end_day]

HK_data = pd.DataFrame(lif.import_HK_data(data_dir, day_folders))

EPOCH_OFFSET_SECONDS = (dt(1970, 1, 1) - dt(1904, 1, 1)).total_seconds()

corrected_timestamps = HK_data['Time_s'] - EPOCH_OFFSET_SECONDS
HK_data['Time_s'] = pd.to_datetime(corrected_timestamps, unit='s')

HK_data['dk_cts_mW-1'] = HK_data['Signal_Dk_Counts'] / HK_data['Laser_Power_PT_1']
HK_data['off_cts_mW-1'] = HK_data['Sig_A_Offline'] / HK_data['Laser_Power_PT_1']
HK_data['dk_cts_norm'] = HK_data['dk_cts_mW-1'] / HK_data['Ref_Dk_Counts']
HK_data['off_cts_norm'] = HK_data['off_cts_mW-1'] / HK_data['Ref_Offline']


# --- set masks for the specific periods of interest ---
test_period_1_mask = (
    HK_data['Time_s'] >= pd.to_datetime(test_period_1_start, dayfirst=True)
) & (
    HK_data['Time_s'] <= pd.to_datetime(test_period_1_end, dayfirst=True)
) 
     
test_period_2_mask = (
    HK_data['Time_s'] >= pd.to_datetime(test_period_2_start, dayfirst=True)
) & (
    HK_data['Time_s'] <= pd.to_datetime(test_period_2_end, dayfirst=True)
)

test_period_1_data = HK_data[test_period_1_mask]
test_period_2_data = HK_data[test_period_2_mask]



for var in plot_vars:
    
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(
        test_period_1_data['Time_s']
        , test_period_1_data[var]
        , label=f'{test_period_1} sig dark counts'
        )
    ax[0].set_xlabel('date_time')
    ax[0].set_ylabel('counts')
    ax[0].legend()
    
    ax[1].plot(
        test_period_2_data['Time_s']
        , test_period_2_data[var]
        , label=f'{test_period_2} sig dark counts'
        , color='red'
        )
    ax[1].set_xlabel('date_time')
    ax[1].set_ylabel('counts')
    ax[1].legend()
    
    fig.suptitle(f"{var} comparison" 
                 f"\nmean ({test_period_1}):{(test_period_1_data[var].replace([np.inf, -np.inf], np.nan).mean()):.3f}"
                 f"\nmean ({test_period_2}):{(test_period_2_data[var].replace([np.inf, -np.inf], np.nan).mean()):.3f}"
                 )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    

if allan_dev == True:
    
    fig, ax = plt.subplots(figsize=(10, 6))
    r = 5  # sample rate in Hz
    t = np.logspace(-1, 3, 100) # tau values from 0.1 to 10000
    
    y_1 = ((test_period_1_data['Sig_A_Online'] - test_period_1_data['Sig_A_Offline'])
           / test_period_1_data['Laser_Power_PT_1']) / test_period_1_data['Ref_Offline']
    y_1_clean = y_1.replace([np.inf, -np.inf], np.nan).dropna().values
    
    if len(y_1_clean) > 0:
        (t2_1, ad_1, _, _) = allantools.oadev(y_1_clean, rate=r, data_type="freq", taus=t)
        ax.loglog(t2_1, ad_1, label=f'{test_period_1} ADEV')
    else:
        ax.text(
            0.5
            , 0.5
            , f"{test_period_1} data is empty/all non-finite."
            , transform=ax[0].transAxes
            , ha='center'
            )
    
    y_2 = ((test_period_2_data['Sig_A_Online'] - test_period_2_data['Sig_A_Offline'])
           / test_period_2_data['Laser_Power_PT_1']) / test_period_2_data['Ref_Offline']
    y_2_clean = y_2.replace([np.inf, -np.inf], np.nan).dropna().values
    
    if len(y_2_clean) > 0:
        (t2_2, ad_2, _, _) = allantools.oadev(y_2_clean, rate=r, data_type="freq", taus=t)
        ax.loglog(t2_2, ad_2, label=f'{test_period_2} ADEV', color='red')
    else:
        ax.text(
            0.5
            , 0.3
            , f"{test_period_2} data is empty/all non-finite."
            , transform=ax[0].transAxes
            , ha='center'
            )
    
    ax.set_title('Allan Deviation Comparison')
    ax.set_xlabel(r'$\tau$ (s)')
    ax.set_ylabel(r'$\sigma_y(\tau)$')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
