# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:44:13 2026

@author: Eve
"""
import lif_functions as lif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
data_dir = (r"E:\boat_SO2_data\test folder")
day_folders = lif.find_day_folders(
    data_dir
    )

MH_HK = lif.import_HK_data(data_dir, day_folders)
start_date = '2024-11-25 00:00:00'
end_date = '2024-11-27 23:00:00'
MAC_TO_EPOCH_OFFSET = 2082844800
MH_HK["Date_time"] = pd.to_datetime(
    MH_HK["Time_s"] - MAC_TO_EPOCH_OFFSET, unit="s")
df_real = pd.DataFrame(MH_HK) 
"""


#-------------------------------------------------------------------------------#
                            #DY195 RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
wind_folder = (r"E:\DY195 FLUX WEATHER DATA\Week 1 to 3 v2")
interuptions_csv_path = r"C:\Users\Eve\Documents\Year 2\Tasmania\phins reprocessing code test\editing_for_the_shift\CARES BOAT interuptions.csv"
underway_file_path = r"C:\Users\Eve\Documents\Year 2\CARES\BOAT DATA_other\Underway_1min.csv"
cloud_fraction_data_path = r"C:\Users\Eve\Documents\Year 2\CARES\cloud fraction (DY195).csv"



"""
fig, j=plt.subplots(1,1)
j.plot(SO2_data["Date_time"], SO2_data["amb_SO2_ppt"])
plt.xlabel("Date_time(UTC)")
plt.ylabel("SO2(ppt)")
plt.ylim(ymax =3000)
w =j.twinx()
w.plot(df_real["Date_time"], df_real["Cal_SO2_MFC_Read"], color = "r", alpha = 0.7)
plt.ylim(ymin =-10, ymax =5.5)
plt.ylabel("Cal_SO2_MFC_Read")
#checking the parts where we were on the other inlet line
start_date = '2025-06-21 07:35:00'
end_date = '2025-06-21 11:24:00'
mask = (SO2_data["Date_time"] >= start_date) & \
       (SO2_data["Date_time"] <= end_date) & \
       np.isfinite(combined_df_final["amb_SO2_ppt"]) & \
       np.isfinite(combined_df_final["CIMSppt"])

df_clean = combined_df_final[mask]

fig, i = plt.subplots(1,1)
lns1 = i.plot(df_clean["Date_time"], df_clean["amb_SO2_ppt"], label = "LIF")
lns2 = i.plot(df_clean["Date_time"], df_clean["CIMSppt"], label = "CIMS")
plt.ylabel("SO2 (ppt)", fontsize = 20)
plt.xlabel("Datetime", fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()
"""
df = pd.read_csv(r"E:\boat_SO2_data\corr flow\1min avg\1 min time sep\v1_DY195_data.txt")

#######------------------------------------------########
#SAVING DATA WHICH STILL HAS THE SPIKES WITHIN IT
#######------------------------------------------########
processed_data_dir = (r"E:\zeroboattests\using a imls for the zero")
#ppt_data = lif.join_txt(processed_data_dir, campaign = "DY195")
SO2_data = lif.SO2_plot_data(processed_data_dir,
                filename = "SO2_data_DY195_07-18_interp_c_k1_zero_imls_removal_v1_1min_avg_20250204", 
                campaign = "DY195", version = "v1")

SO2_data_spikes = lif.DY195_flagged_periods_for_keeping_spikes(interuptions_csv_path, SO2_data)
SO2_restart_removed = lif.restart_removed(SO2_data_spikes)
filename ="1min c (k1)  zeros (imls 40minute window)  no end removal(07 to 18)"
lif.save_to_csv(processed_data_dir, filename, SO2_restart_removed)
#now spat out the datafile, so we can plot it up with the CIMS dat

#comparison periods retained only
SO2_comparison_periods = lif.DY195_comparison_periods(interuptions_csv_path, df)
SO2_restart_removed = lif.restart_removed(SO2_comparison_periods)
filename ="1min avg cruise data, comp periods only retained"
lif.save_to_csv(processed_data_dir, filename, SO2_restart_removed)






filtered_DY195_SO2_data = lif.DY195_flagged_periods(interuptions_csv_path,SO2_data)
underway_data = lif.load_ship_track(underway_file_path, filtered_DY195_SO2_data)

SO2_1min_in_sector = lif.DY195_in_sector(underway_data, filtered_DY195_SO2_data)
SO2_restart_removed = lif.restart_removed(SO2_1min_in_sector)
filename = "SO2_1min_avg_in_sector_interuptions_removed_v1_20260123"
lif.save_to_csv(
    processed_data_dir, filename, SO2_restart_removed
    )
#now have filtered 1 min data. THEN we can do the resample to 5min
SO2_restart_removed = SO2_data
lif.DY195_in_sector_diurnal_1min(SO2_restart_removed)
#getting it as a 5 min averaged plot!
SO2_5min_in_sector = lif.DY195_in_sector_5min(underway_data, SO2_1min_in_sector)
lif.DY195_in_sector_diurnal_5min(SO2_5min_in_sector)
#cloud data from Ming
lif.cloud_fraction(cloud_fraction_data_path, SO2_1min_in_sector)
#winds = lif.load_all_winds_DY195(wind_folder)


"""
pos_df = underway_data
# === Plot ship track and EEZ ===
import matplotlib.dates as mdates

# 1. Convert pandas datetimes to Matplotlib dates
pos_df["time_1min"] = pd.to_datetime(pos_df["time_1min"])
map_dates = mdates.date2num(pos_df["time_1min"])

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_facecolor('lightcyan')

# 1. Plot the track and scatter as before
map_dates = mdates.date2num(pos_df["time_1min"])
sc = ax.scatter(pos_df["lon_deg_1min"], pos_df["lat_deg_1min"],
                c=map_dates, cmap="viridis", s=40, zorder=3)

# 2. Add Time Labels at specific intervals (e.g., every 60 points = every hour)
# Change the '60' to a larger number if the map is too crowded
interval = 60 

for i in range(0, len(pos_df), interval):
    label_time = pos_df["time_1min"].iloc[i].strftime('%d-%m %H:%M')
    ax.annotate(
        label_time, 
        (pos_df["lon_deg_1min"].iloc[i], pos_df["lat_deg_1min"].iloc[i]),
        textcoords="offset points", 
        xytext=(5,5), # Small offset so text isn't directly on top of the point
        fontsize=9,
        fontweight='bold',
        color='black',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )

# 3. Standard Colorbar setup
cbar = plt.colorbar(sc, ax=ax)
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
cbar.set_label("Date/Time (UTC-1)", fontsize=18)

# Start/End markers
ax.scatter(pos_df["lon_deg_1min"].iloc[0], pos_df["lat_deg_1min"].iloc[0],
           color='red', marker='x', s=100, label='Start', zorder=5)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Ship Track with Time Stamps")
ax.grid(True)
plt.show()
"""




#-------------------------------------------------------------------------------#
                            #GRIMSAF RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
processed_data_dir = (r"E:\SOOZ\GRIMSAF\data\v1_processed_data_file")
SO2_data = lif.SO2_plot_data_GRIMSAF(processed_data_dir,
                    filename, 
                    campaign = "GRIMSAF", version = "v1")
#had to chnage the resample function a little bit and then also the one where we assign the ambient
#value as I removed Task =2 and left everything else in!

SO2_data["Date_time"] = pd.to_datetime(SO2_data["Date_time"])

start_date = "2025-09-24 17:57:00"
end_date = "2025-09-24 17:59:00"
SO2_data_mean = SO2_data.loc[
    (SO2_data['Date_time'] >= start_date) & 
    (SO2_data['Date_time'] <= end_date),"amb_SO2_ppt"].mean()
print(SO2_data_mean)

fig, flow_amb = plt.subplots(1,1)
flow_amb.plot(SO2_data["Date_time"], SO2_data["amb_SO2_ppt"])
r = flow_amb.twinx()
r.plot(cts_data_zeroed["Date_time"], cts_data_zeroed["Task"], color = "purple")

#just used the housekeeping for this (so used the hK quick look functions/ code)
start_date = "2025-09-22 15:12:00"
end_date = "2025-09-22 17:08:00"
Flows = cts_data_ref_norm_sec.loc[
    (cts_data_ref_norm_sec['Date_time'] >= start_date) & 
    (cts_data_ref_norm_sec['Date_time'] <= end_date)].mean()
print(Flows["Cell_Flow"], Flows["ZA_SB_MFC_Read"], Flows["Cal_ZA_MFC_Read"])

# cal_data_GRIMSAF = pd.read_csv("E:\SOOZ\GRIMSAF\data\sig_cal_data.txt", header =0)
# average_cal_factor = np.mean(cal_data_GRIMSAF["slope"])
# GRIMSAF_data_10Hz = cts_data_zeroed["sig_diff_cts_ref_norm_zero_corr"] / average_cal_factor

# GRIMSAF_data_10Hz = GRIMSAF_data_10Hz.to_frame()

# plt.plot(GRIMSAF_data_10Hz.index, GRIMSAF_data_10Hz["sig_diff_cts_ref_norm_zero_corr"])
# print(GRIMSAF_data_10Hz["sig_diff_cts_ref_norm_zero_corr"])



#-------------------------------------------------------------------------------#
                            #MACE HEAD RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#


#MACE HEAD
processed_data_dir = (r"E:\LOKISO2\MACE_HEAD\data\v1_processed_data_file")
SO2_data = lif.SO2_plot_data(processed_data_dir,
                    filename, 
                    campaign = "MACE_HEAD", version = "v1")   

# want the baseline data to be imported in and used
#using column 10
# windrose?
#dirunals
#can we make the diurnals just be used for all of the data? and they just 

#-------------------------------------------------------------------------------#
                            #TASMANIA RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#

processed_data_dir = (r"")
SO2_data = lif.SO2_plot_data(processed_data_dir,
                    filename, 
                    campaign = "TASMANIA", version = "v1") 
#baseline data
#winds
#irradiance to be with the so2?
#temperature be a good one?

#need to have a look at the zeros over the course of the campgain