import lif_functions as lif
import matplotlib.pyplot as plt
import pandas as pd
# --------------------- Variables to manually input --------------------------

molecule = 'SO2'                                                                 # NO (even for multi-channel NOx) or SO2
channels = {'sig':'SO2'}                                                         # dict of all sig channels
cal_task = 2                                                                    # Task number associated with regular cals
R2_limit = 0.8                                                               # Task number associated with BLC cals
BLC = False                                                                     # Boolean indicator of whether BLC needs analysing
plot = True                                                                     # diagnostics plots at various analysis stages
averaging = '10S' # 5 min                                                            # averaging for the final output file
filename = 'SO2_GRIMSAF_v1_20260105'                                 # filename for the resampled MR output
cal_cylinder_conc = 5100                                                        # in ppb
pre_taskswitch = 1#300                                                            # data points before task switch to ignore
post_taskswitch = 400                                                           # data points after task switch to ignore
pre_peakfind = 20                                                               # data points before ref_cts_diff drop to ignore
post_peakfind = 60#200                                                             # data points after ref_cts_diff drop to ignore
#ref_cts_diff_limit = 188000  #DY195                                                    # lower limit ref_diff_cts_norm
ref_cts_diff_limit = 150000  #GRIMSAF   
#DATA DIRECTORIES
#data_dir = (r"E:\boat_SO2_data")
data_dir = (r"E:\SOOZ\GRIMSAF\data")
wind_folder = (r"E:\DY195 FLUX WEATHER DATA\Week 1 to 3 v2")
# ----------------------------------------------------------------------------


day_folders = lif.find_day_folders(
    data_dir
    )
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )
# cts_data = lif.CARES_NO_ref_correction(
#     cts_data
#     )
cts_data_ref_norm = lif.ref_normalise(
    cts_data, channels=channels
    )

#DY195 sectioning!
cts_data_ref_norm_sec = cts_data_ref_norm.copy()
cts_data_ref_norm_sec['Date_time'] = pd.to_datetime(cts_data_ref_norm_sec['Date_time'])
plt.plot(cts_data_ref_norm_sec["Date_time"], cts_data_ref_norm_sec["ref_diff_cts_norm"])
plt.plot(cts_data_ref_norm_sec["Date_time"], cts_data_ref_norm_sec["sig_diff_cts_norm"])

# start_date = "2025-06-07 12:00:00"
# end_date = "2025-06-25 23:00:00"
# cts_data_ref_norm_sec = cts_data_ref_norm_sec[
#     (cts_data_ref_norm_sec['Date_time'] >= start_date) & 
#     (cts_data_ref_norm_sec['Date_time'] <= end_date)
# ]

#back to the code!
cts_data_flagged = lif.set_flags(
    cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind
    , post_peakfind, ref_cts_diff_limit
    )
cts_data_zeroed = lif.zero_correct_average(
    cts_data_flagged, channels=channels, plot=plot
    )
cts_data_zeroed["Date_time"] = pd.to_datetime(cts_data_zeroed["Date_time"])
plt.plot(cts_data_zeroed["Date_time"], cts_data_zeroed["sig_diff_cts_norm"])
plt.plot(cts_data_zeroed["Date_time"], cts_data_zeroed["ref_diff_cts_norm"])

#EVE PIECE OF MIND CHECK
cts_data_single_point_plotting = lif.cal_single_point(
    cts_data_flagged, channels=channels, plot = plot
    )
#back to the actual code!
lif.analyse_cals(
    cts_data_zeroed, data_dir, channels=channels, molecule=molecule
    , cal_cylinder_conc=cal_cylinder_conc, plot=plot, save_csv=True
    , cal_task=cal_task , threshold = R2_limit
)
MR_data = lif.apply_cals_average(
    data_dir, cts_data_zeroed, channels=channels, BLC=BLC, R2_limit = R2_limit
    )
MR_data_resampled = lif.resample_data(
    MR_data, averaging
    )
lif.save_to_csv(
    data_dir, filename, MR_data_resampled
    )


#DY195 RELEVANT FUNCTIONS
"""
processed_data_dir = (r"E:\boat_SO2_data\SO2_txt_ppt_files_1min")
#ppt_data = lif.join_txt(processed_data_dir, campaign = "DY195")
SO2_data = lif.SO2_plot_data_DY195(processed_data_dir,
                    filename = "v1_DY195_data", 
                    campaign = "DY195", version = "v1")

interuptions_csv_path = "CARES BOAT interuptions to data collection - Sheet1.csv"
filtered_DY195_SO2_data = lif.DY195_flagged_periods(interuptions_csv_path,SO2_data)

underway_file_path = r"C:\Users\Eve\Documents\Year 2\CARES\BOAT DATA_other\Underway_1min.csv"

underway_data = lif.load_ship_track(underway_file_path, filtered_DY195_SO2_data)
SO2_1min_in_sector = lif.DY195_in_sector(underway_data, filtered_DY195_SO2_data)
#now have filtered 1 min data. THEN we can do the resample to 5min
lif.DY195_in_sector_diurnal_1min(SO2_1min_in_sector)
#getting it as a 5 min averaged plot!
SO2_5min_in_sector = lif.DY195_in_sector_5min(underway_data, SO2_1min_in_sector)
lif.DY195_in_sector_diurnal_5min(SO2_5min_in_sector)

#cloud data from Ming
cloud_fraction_data_path = r"C:\Users\Eve\Documents\Year 2\CARES\cloud fraction (DY195).csv"
lif.cloud_fraction(cloud_fraction_data_path, SO2_1min_in_sector)
#winds = lif.load_all_winds_DY195(wind_folder)
"""

#GRIMSAF data analysis
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