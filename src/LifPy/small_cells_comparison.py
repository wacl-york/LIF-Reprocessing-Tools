# =============================================================================
# start_time = 
# end_time = 
# items_to_visualise = []
# =============================================================================

# read the processed files 
# subset the data for the times indicated as the test period 
import allantools

import numpy as np
import pandas as pd
import lif_functions as lif   
import matplotlib.pyplot as plt

from datetime import datetime as dt 
 
data_dir = (r'C:\Users\pp835\OneDrive - University of York\Documents\Data Analysis\CARES\Post campaign testing\Data')

day_folders = lif.find_day_folders(data_dir)

HK_data = pd.DataFrame(lif.import_HK_data(data_dir, day_folders))

EPOCH_OFFSET_SECONDS = (
    dt(1970, 1, 1) - dt(1904, 1, 1)
).total_seconds()

corrected_timestamps = HK_data['Time_s'] - EPOCH_OFFSET_SECONDS
HK_data['Time_s'] = pd.to_datetime(corrected_timestamps, unit='s')

HK_data['dk_cts_norm'] = (HK_data['Signal_Dk_Counts'] / HK_data['Laser_Power_PT_1']) / HK_data['Ref_Dk_Counts']
HK_data['off_cts_norm'] = (HK_data['Sig_A_Offline'] / HK_data['Laser_Power_PT_1']) / HK_data['Ref_Offline']



big_cell_test_mask = (
    HK_data['Time_s'] >= pd.to_datetime('04/11/2025 13:30', dayfirst=True)
) & (
    HK_data['Time_s'] <= pd.to_datetime('04/11/2025 16:00', dayfirst=True)
) 
small_cell_test_mask = (
    HK_data['Time_s'] >= pd.to_datetime('13/11/2025 10:15', dayfirst=True)
) & (
    HK_data['Time_s'] <= pd.to_datetime('13/11/2025 13:15', dayfirst=True)
)

big_cell_data = HK_data[big_cell_test_mask]
small_cell_data = HK_data[small_cell_test_mask]






fig, ax = plt.subplots(2, 1, figsize=(10,10))
ax[0].plot(big_cell_data['Time_s'], big_cell_data['dk_cts_norm'], label='big cell sig dark counts')
ax[0].set_xlabel('date_time')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].plot(small_cell_data['Time_s'], small_cell_data['dk_cts_norm'], label='small cell sig dark counts', color='red')
ax[1].set_xlabel('date_time')
ax[1].set_ylabel('counts')
ax[1].legend()

fig.suptitle(f"ref normalised dark count comparison" 
             f"\nmean dk cts big cell:{(big_cell_data['dk_cts_norm'].mean()):.3f}"
             f"\nmean dk cts small cell:{(small_cell_data['dk_cts_norm'].mean()):.3f}"
             )
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





fig, ax = plt.subplots(2, 1, figsize=(10,10))
ax[0].plot(big_cell_data['Time_s'], big_cell_data['Signal_Dk_Counts'], label='big cell sig dark counts')
ax[0].set_xlabel('date_time')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].plot(small_cell_data['Time_s'], small_cell_data['Signal_Dk_Counts'], label='small cell sig dark counts', color='red')
ax[1].set_xlabel('date_time')
ax[1].set_ylabel('counts')
ax[1].legend()

fig.suptitle(f"raw dark count comparison" 
             f"\nmean dk cts big cell:{(big_cell_data['Signal_Dk_Counts'].mean()):.3f}"
             f"\nmean dk cts small cell:{(small_cell_data['Signal_Dk_Counts'].mean()):.3f}"
             )
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





fig, ax = plt.subplots(2, 1, figsize=(10,10))
ax[0].plot(big_cell_data['Time_s'], big_cell_data['off_cts_norm'], label='big cell offline counts')
ax[0].set_xlabel('date_time')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].plot(small_cell_data['Time_s'], small_cell_data['off_cts_norm'], label='small cell offline counts', color='red')
ax[1].set_xlabel('date_time')
ax[1].set_ylabel('counts')
ax[1].legend()

fig.suptitle(f"ref normalised signal offline count comparison" 
             f"\nmean off cts big cell:{(big_cell_data['off_cts_norm'].mean()):.3f}"
             f"\nmean off cts small cell:{(small_cell_data['off_cts_norm'].mean()):.3f}"
             )
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




# --- Setup Figure ---
fig, ax = plt.subplots(figsize=(10, 6))
r = 5  # sample rate in Hz
t = np.logspace(-1, 3, 100) # tau values from 0.1 to 10000

# --- 1. Big Cell Data Calculation & Plotting ---
y_big = ((big_cell_data['Sig_A_Online'] - big_cell_data['Sig_A_Offline'])/ big_cell_data['Laser_Power_PT_1']) / big_cell_data['Ref_Offline']
# **CRITICAL FIX 1: Drop NaN/Inf and Convert to a clean numpy array**
y_big_clean = y_big.replace([np.inf, -np.inf], np.nan).dropna().values

if len(y_big_clean) > 0:
    (t2_big, ad_big, _, _) = allantools.oadev(y_big_clean, rate=r, data_type="freq", taus=t)
    ax.loglog(t2_big, ad_big, label='Big Cell ADEV')
else:
    # Handle empty data case
    ax.text(0.5, 0.5, "Big Cell data is empty/all non-finite.", transform=ax[0].transAxes, ha='center')


# --- 2. Small Cell Data Calculation & Plotting ---
y_small = ((small_cell_data['Sig_A_Online'] - small_cell_data['Sig_A_Offline'])/ small_cell_data['Laser_Power_PT_1']) / small_cell_data['Ref_Offline']
# **CRITICAL FIX 2: Drop NaN/Inf and Convert to a clean numpy array**
y_small_clean = y_small.replace([np.inf, -np.inf], np.nan).dropna().values

if len(y_small_clean) > 0:
    (t2_small, ad_small, _, _) = allantools.oadev(y_small_clean, rate=r, data_type="freq", taus=t)
    ax.loglog(t2_small, ad_small, label='Small Cell ADEV', color='red')
else:
    ax.text(0.5, 0.3, "Small Cell data is empty/all non-finite.", transform=ax[0].transAxes, ha='center')

# --- Final Plot Enhancements for ax[0] ---
ax.set_title('Allan Deviation Comparison')
ax.set_xlabel(r'$\tau$ (s)')
ax.set_ylabel(r'$\sigma_y(\tau)$')
ax.legend()

plt.tight_layout()
plt.show()


# =============================================================================
# fig, ax = plt.subplots(2, 1, figsize=(10,10))  
# 
# t = np.logspace(-1, 3, 100)  # tau values from 0.1 to 10000
# y = big_cell_data['Sig_A_Online'] - big_cell_data['Sig_A_Offline'] 
# r = 5  # sample rate in Hz of the input data
# (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # Compute the overlapping ADEV
# ax[0].loglog(t2, ad)
# 
# t = np.logspace(-1, 3, 100)  # tau values from 0.1 to 10000
# y = small_cell_data['Sig_A_Online'] - small_cell_data['Sig_A_Offline'] 
# r = 5  # sample rate in Hz of the input data
# (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # Compute the overlapping ADEV
# ax[1].loglog(t2, ad)
# 
# plt.show()
# =============================================================================
