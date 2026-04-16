# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:44:13 2026

@author: Eve
"""
import lif_functions as lif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#join together the data if been split into a few sections!
processed_data_dir = (r"E:\V1_data\MACE HEAD")
ppt_data = lif.join_txt(processed_data_dir, campaign = "MACE_HEAD")


#-------------------------------------------------------------------------------#
                            #DY195 RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
wind_folder = (r"E:\DY195 FLUX WEATHER DATA\Week 1 to 3 v2")
interuptions_csv_path = r"C:\Users\Eve\Documents\Year 2\Tasmania\phins reprocessing code test\editing_for_the_shift\CARES BOAT interuptions.csv"
underway_file_path = r"C:\Users\Eve\Documents\Year 2\CARES\BOAT DATA_other\Underway_1min.csv"
cloud_fraction_data_path = r"C:\Users\Eve\Documents\Year 2\CARES\cloud fraction (DY195).csv"
data_dir = (r"E:\boat_SO2_data")
day_folders = lif.find_day_folders(
    data_dir
    )
processed_data_dir = (r"E:\V1_data\DY195")

#ppt_data = lif.join_txt(processed_data_dir, campaign = "DY195")
SO2_data = lif.SO2_plot_data_with_resample(processed_data_dir,
                filename = "v1_DY195_data", 
                campaign = "DY195", version = "v1", resample = True, averaging = "1 min", plot=True)
restart_removed_SO2_data = lif.remove_restart(SO2_data, data_dir, day_folders)

#filtering for the data for my flags
filtered_DY195_SO2_data = lif.DY195_flagged_periods(interuptions_csv_path,restart_removed_SO2_data)

#keep the flagged data and then flag for the periods which we trust.
#Leave it as it is. DO not do in sector.
flagged_v1_data = lif.flag_data(filtered_DY195_SO2_data, plot = True, version = "v1", campagin_list = "DY195")
filename = "2026_04_14_DY195_v1"
lif.save_to_csv(processed_data_dir, filename, flagged_v1_data)
#filtering for the data only wihtin the clean sector period
underway_data = lif.load_ship_track(underway_file_path, flagged_v1_data)
SO2_1min_in_sector = lif.DY195_in_sector(underway_data, flagged_v1_data)
lif.DY195_in_sector_diurnal_1min(SO2_1min_in_sector)





#for if we want to keep the spikes in and do a comparison if the overall data
SO2_data_spikes = lif.DY195_flagged_periods_for_keeping_spikes(interuptions_csv_path, SO2_data)
SO2_restart_removed = lif.restart_removed(SO2_data_spikes)
filename ="1min with the spikes in"
lif.save_to_csv(processed_data_dir, filename, SO2_restart_removed)
#now spat out the datafile, so we can plot it up with the CIMS dat

#comparison periods retained only
SO2_comparison_periods = lif.DY195_comparison_periods(interuptions_csv_path, df)

#SO2_restart_removed = lif.restart_removed(SO2_1min_in_sector)
filename = "SO2_1min_avg_in_sector_interuptions_removed_v1_20260320"
lif.save_to_csv(
    processed_data_dir, filename, SO2_restart_removed
    )
data = pd.read_csv(r"C:\Users\Eve\Documents\Year 3\DATA_ANALYSIS\DY195\2026_03_20\SO2_1min_avg_in_sector_interuptions_removed_v1_20260320.txt")
#now have filtered 1 min data. THEN we can do the resample to 5min
data["Date_time"] = pd.to_datetime(data["Date_time"])

fig, i = plt.subplots(1,1)
i.plot(data["Date_time"], data["amb_SO2_ppt"])
plt.ylabel("SO2 (ppt)", fontsize = 20)
plt.xlabel("Datetime (UTC)", fontsize = 20)
plt.title("1min averaged SO2 during DY195, interuptions removed, in sector", fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

SO2_restart_removed = SO2_data

#tetsing of meterological parameters with SO2
winds = lif.DY195_load_all_winds(wind_folder)
testing = lif.DY195_meterological_parameters_testing(data, underway_file_path)






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
data_dir = (r"E:\LOKISO2\MACE HEAD\MACE_HEAD_SO2")
day_folders = lif.find_day_folders(
    data_dir
    )
processed_data_dir = (r"G:\My Drive\SO2_instrument\Data\CARES\MACE HEAD\v1_processed_MH_data")
SO2_data = lif.SO2_plot_data(processed_data_dir,
                    filename = "2026_04_14_MACE_HEAD_v1", 
                    campaign = "MACE_HEAD", version = "v1") 
#remove the restart from the data
restart_removed_SO2_data = lif.remove_restart(SO2_data, data_dir, day_folders)
#flags for interuptiosn to the data!
MH_interuptions_path = r"E:\LOKISO2\MACE HEAD\MACE_HEAD_INTERRUPTIONS_FLAGGER.csv" 
SO2_data_flagged = lif.MH_flagged_periods(MH_interuptions_path, restart_removed_SO2_data)

flagged_v1_data = lif.flag_data(SO2_data_flagged, plot = True, version = "v1", campagin_list = "MACE_HEAD")
filename = "2026_04_14_MACE_HEAD_v1"
lif.save_to_csv(processed_data_dir, filename, flagged_v1_data)

#defining the baseline using Alex's baseline defs
data_dir_MH_base = r"E:\LOKISO2\MACE HEAD\Mace Head other data"
SO2_baseline = lif.MH_baseline_flagger(data_dir_MH_base, SO2_data_flagged)

#make dirunals in the baseline sector
SO2_baseline_dirunal = lif.MH_diurnal(SO2_baseline)

#looking at the met params, may need to do it for when we only flag for SO2 inst error and then
#see if there are other interuptiosn past this.
#MH interuptiosn file is sparse cause the notes are so should be just what I can find
#and that should be instrument stuff and nothign else.
met_data_dir = r"E:\LOKISO2\MACE HEAD\Mace Head other data\VSys_dtLog_May2025.csv"
SO2_met_data = lif.MH_met_data(met_data_dir, SO2_data_flagged)

#d.to_csv("MH_data_v1_first_half.csv")
winds = pd.read_csv(r"E:\LOKISO2\MACE HEAD\Mace Head other data\Wind data.csv")
winds["Date_time"] = pd.to_datetime(winds["Date_time"])

fig, i = plt.subplots(2,1, sharex = True)
i[0].plot(winds["Date_time"],winds["wind direction"])
i[1].plot(SO2_data_flagged["Date_time"], SO2_data_flagged["amb_SO2_ppt"])
# want the baseline data to be imported in and used
#using column 10
# windrose?
#dirunals
#can we make the diurnals just be used for all of the data? and they just 

#looking at the DMS and MeSH data
VOCUS_PTR = pd.read_csv(r"E:\V1_data\MACE HEAD\DMS and MeSH\Masterfile.csv", header = 0)
VOCUS_PTR["Date_and_Time_(UTC)"] = pd.to_datetime(VOCUS_PTR["Date_and_Time_(UTC)"], dayfirst = True, format = "mixed")

fig, g = plt.subplots(3,1, sharex = True)
g[0].plot(VOCUS_PTR["Date_and_Time_(UTC)"],VOCUS_PTR["DMS_(ppt)"])
g[0].set_ylabel("DMS (ppt)", fontsize = 20)
g[1].plot(VOCUS_PTR["Date_and_Time_(UTC)"],VOCUS_PTR["DMSO2_(ppt)"])
g[1].set_ylabel("MeSH (ppt)", fontsize = 20)
g[2].plot(flagged_v1_data["Date_time"], flagged_v1_data["amb_SO2_ppt"])
g[2].set_ylabel("SO2 (ppt)", fontsize = 20)




#-------------------------------------------------------------------------------#
                            #TASMANIA RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
import lif_functions as lif
#read in the processed data dir
processed_data_dir = (r"E:\V1_data\TASMANIA\2026-04-13 checking")
ppt_data = lif.join_txt(processed_data_dir, campaign = "TASMANIA")

data_dir = (r"G:\My Drive\SO2_instrument\Data\Tasmania 2024\NOV-JAN_09")
day_folders = lif.find_day_folders(
    data_dir
    )
#plotting out the timeseries of the correct data
SO2_data = lif.SO2_plot_data(processed_data_dir,
                filename = "v1_TASMANIA_data", 
                campaign = "TASMANIA", version = "v1")
restart_removed_SO2_data = lif.remove_restart(SO2_data, data_dir, day_folders)

#interruptions flags to a v1 of the data!
TAS_interuptions_path = (r"C:\Users\Eve\Documents\Year 2\Tasmania\TASMANIA_CAMPGAIN_DATA\TAS_INTERRUPTIONS.csv")
SO2_data_flagged = lif.Tasmania_flagged_periods(TAS_interuptions_path, SO2_data)


flagged_v1_data = lif.flag_data(SO2_data_flagged, plot = True, version = "v1", campagin_list = "TASMANIA")
filename = "2026_04_14_TASMANIA_v1"
lif.save_to_csv(processed_data_dir, filename, flagged_v1_data)

data = pd.read_csv(r"G:\My Drive\SO2_instrument\Data\Tasmania 2024\v1_data\2026_04_14_TASMANIA_v1.txt")

data["Date_time"] = pd.to_datetime(data["Date_time"])

mask = data["flag"] ==0

data = data[mask]

fig, i = plt.subplots(1,1)
i.plot(data["Date_time"], data["amb_SO2_ppt"])

SO2_to_aus_time = lif.timezone_conversion_AEDT(flagged_v1_data)
diurnals = lif.TAS_in_sector_diurnal_1min(SO2_to_aus_time)
filename = "20260331_SO2_JAN_MAR_2025_flags_removed_all_sectors"
lif.save_to_csv_TAS(processed_data_dir , filename, flagged_data)
#baseline data: using the BL3 flag for the baseline data
T_baseline_flagger_path = (r"C:\Users\Eve\Documents\Year 2\Tasmania\Diurnal\cgbaps-pops-2025-01_edit.csv")
baseline_SO2_data = lif.Tas_baseline(T_baseline_flagger_path, flagged_v1_data)
baseline_aedt = lif.timezone_conversion_AEDT(baseline_SO2_data)
#dirunal for the basline period
diurnals = lif.TAS_in_sector_diurnal_1min(baseline_aedt)

#winds
TAS_winds_path = (r"C:\Users\Eve\Documents\Year 2\Tasmania\TASMANIA_CAMPGAIN_DATA\winds_data_Nov_to_March.csv")
met_data_dir = (r"C:\Users\Eve\Documents\Year 2\Tasmania\TASMANIA_CAMPGAIN_DATA\met_data_Nov_to_March.csv")
wind_rose = lif.TAS_windrose(flagged_data, met_data_dir, TAS_winds_path)

#irradiance to be with the so2?
TAS_solar_path = (r)

#met data plotting against them. Think that they are all in UTC!
met_so2_comp = lif.TAS_met_data(met_data_dir, flagged_data)


flagged_v1_data = lif.flag_data(SO2_data, plot = True, version = "v1", campagin_list = "TASMANIA")
