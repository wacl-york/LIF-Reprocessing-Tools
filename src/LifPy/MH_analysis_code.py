# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:56:38 2026

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
                            #MACE HEAD RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
data_dir = (r"E:\LOKISO2\MACE HEAD\MACE_HEAD_SO2")
day_folders = lif.find_day_folders(
    data_dir
    )

processed_data_dir = (r"E:\V1_data\MACE HEAD\LIFSO2")
SO2_v1_data = lif.SO2_plot_data(processed_data_dir,
                    filename = "2026_04_15_MACE_HEAD_v1", 
                    campaign = "MACE_HEAD", version = "v1") 
"""
SO2_data_flagged_v1_data["Date_time"] = pd.to_datetime(SO2_data_flagged_v1_data["Date_time"])
plt.rcParams['svg.fonttype'] = 'path'
import matplotlib.ticker as ticker
fig, i = plt.subplots(1,1, figsize=(14, 7))
i.plot(SO2_data_flagged_v1_data["Date_time"], SO2_data_flagged_v1_data["amb_SO2_ppt"])
plt.ylabel("$SO_2$ (ppt)", fontsize = 16)
plt.xlabel("Date_time", fontsize = 16)
plt.ylim(ymax = 500, ymin = -50)
i.xaxis.set_major_locator(ticker.MaxNLocator(10))

# 3. Rotate and align
plt.xticks(rotation=0, fontsize=16)
plt.yticks(fontsize=16)

plt.ylabel("$SO_2$ (ppt)", fontsize=16)
plt.xlabel("Date_time", fontsize=16)

# 4. Save with tight boundaries
plt.savefig("SO2_Mace_Head.svg", format="svg", bbox_inches='tight')
plt.savefig("SO2_Mace_Head.pdf", format="pdf", bbox_inches='tight')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox

# --- MANDATORY SETTINGS ---
plt.rcParams['svg.fonttype'] = 'path'  # Turn letters into solid shapes
plt.rcParams['pdf.fonttype'] = 42      # Backup for PDF export

# 1. Define a fixed size (Inches)
fig_width, fig_height = 14, 8
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# 2. Hard-code the plot area (Left, Bottom, Width, Height as decimals of 1.0)
# This prevents the poster software from moving the axis
ax.set_position([0.15, 0.2, 0.75, 0.7]) 

ax.plot(SO2_data_flagged_v1_data["Date_time"], SO2_data_flagged_v1_data["amb_SO2_ppt"])

# 3. Force fixed spacing
ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
plt.xticks(rotation=0, fontsize=16)
plt.yticks( fontsize = 16)

plt.ylabel("$SO_2$ (ppt)", fontsize=16)
plt.xlabel("Datetime (UTC)", fontsize=16)
plt.ylim(ymax = 250, ymin= -50)

# 4. THE CRITICAL STEP: Save with a fixed bounding box
# We manually define the box so the software can't "fold" it.
full_bbox = Bbox.from_bounds(0, 0, fig_width, fig_height)

plt.savefig("E:\V1_data\MACE HEAD\Mace_Head.svg", 
            format="svg", 
            bbox_inches=full_bbox, 
            pad_inches=0)

# 5. Backup: Save as a High-Res PDF 
# (Design software usually handles PDF vectors better than SVG)
plt.savefig("E:\V1_data\MACE HEAD\Mace_Head.pdf", bbox_inches=full_bbox)
"""
#looking at the different wind directions
wind_direction = "270 to 360 deg"
lif.wind_sector_diurnals(SO2_v1_data, wind_direction, beg_dir = 270, end_dir = 360)
lif.wind_sector_diurnals_SO2_only(SO2_v1_data)

#########################################################
                    #BASELINE
#########################################################

#defining the baseline using Alex's baseline defs
data_dir_MH_base = r"E:\LOKISO2\MACE HEAD\Mace Head other data"
SO2_baseline = lif.MH_baseline_flagger(data_dir_MH_base, SO2_v1_data, VOCUS = True)
filename = "SO2_baseline_Alex"
#lif.save_to_csv(processed_data_dir, filename, SO2_baseline)

#using the MH criteria, but subbing the back trajs for Radon
SO2_baseline1 = lif.MACE_HEAD_baseline_criteria(SO2_v1_data)
filename = "SO2_baseline_MH_and_Radon"
#lif.save_to_csv(processed_data_dir, filename, SO2_baseline1)

#MH baseline defintion
SO2_baseline2 = lif.baseline_from_masterfile(SO2_v1_data)
filename = "SO2_baseline_MH_from_MASTERFILE"
lif.save_to_csv(processed_data_dir, filename, SO2_baseline1)

SO2_baseline["Date_time"] = pd.to_datetime(SO2_baseline["Date_time"])
SO2_baseline1["Date_time"] = pd.to_datetime(SO2_baseline1["Date_time"])
SO2_baseline2["Date_time"] = pd.to_datetime(SO2_baseline2["Date_time"])

fig, o = plt.subplots(1,1)
o.plot(SO2_baseline["Date_time"], SO2_baseline["amb_SO2_ppt"], color = "r", alpha = 0.6, marker = 11, markersize = 3, label = "Model baseline filtering")
o.plot(SO2_baseline1["Date_time"], SO2_baseline1["amb_SO2_ppt"], color = "b", alpha = 0.8, marker = "d", markersize = 3, label = "MH (excl. back traj) + Radon < 0.1Bq")
o.plot(SO2_baseline2["Date_time"], SO2_baseline2["amb_SO2_ppt"], color = "orange", alpha = 0.8, marker = "*", markersize =3,  label = "MH baseline def (had to rebuild the clean def)")
plt.ylabel("$SO_2$ (ppt)", fontsize = 20)
plt.xlabel("Datetime (UTC)", fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend()


#sectioning them into different periods to look at different periods.
beg_sect = lif.time_frame(SO2_v1_data,start_date = "2025-05-31 15:30:00", 
                                                  end_date = "2025-06-13 14:45:00")
end_sect = lif.time_frame(SO2_v1_data,start_date = "2025-06-13 14:45:00", 
                                                  end_date = "2025-06-25 23:00:00")
lif.MH_diurnal_all_params(SO2_baseline)
lif.MH_diurnal_all_params(SO2_baseline1)
lif.MH_diurnal_all_params(SO2_baseline2) # still need to try and get this one to work!
lif.MH_diurnal_all_params(beg_sect)
lif.MH_diurnal_all_params(end_sect)

#make dirunals in the baseline sector
SO2_baseline_dirunal = lif.MH_diurnal_all_params(SO2_baseline)

SO2_baseline["Date_time"] = pd.to_datetime(SO2_baseline["Date_time"])
SO2_baseline1["Date_time"] = pd.to_datetime(SO2_baseline1["Date_time"])

fig, o = plt.subplots(1,1)
o.plot(SO2_baseline["Date_time"], SO2_baseline["amb_SO2_ppt"], color = "r", alpha = 0.6, label = "Model baseline filtering")
o.plot(SO2_baseline1["Date_time"], SO2_baseline1["amb_SO2_ppt"], color = "b", alpha = 0.8, label = "MH (excl. back traj) + Radon < 0.1Bq")
plt.ylabel("$SO_2$ (ppt)", fontsize = 20)
plt.xlabel("Dattime (UTC)", fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend()



##############################################
            #SO2:xx ratios
##############################################

#SO2 to DMS ratio
lif.ratio_calculation(SO2_v1_data)
lif.ratio_calculation(beg_sect)
lif.ratio_calculation(end_sect)

#SO2 ratios, in the baseline
lif.ratio_calculation(SO2_baseline)
lif.ratio_calculation(beg_sect)
lif.ratio_calculation(end_sect)


#looking at the met params, may need to do it for when we only flag for SO2 inst error and then
#see if there are other interuptiosn past this.
#MH interuptiosn file is sparse cause the notes are so should be just what I can find
#and that should be instrument stuff and nothign else.
met_data_dir = r"E:\LOKISO2\MACE HEAD\Mace Head other data\VSys_dtLog_May2025.csv"
SO2_met_data = lif.MH_met_data(met_data_dir, SO2_v1_data, VOCUS= True)


# want the baseline data to be imported in and used
#using column 10
# windrose?
#dirunals
#can we make the diurnals just be used for all of the data? and they just 

#looking at the DMS and MeSH data
VOCUS_PTR = pd.read_csv(r"E:\V1_data\MACE HEAD\DMS and MeSH\Masterfile.csv", header = 0)
VOCUS_PTR["Date_and_Time_(UTC)"] = pd.to_datetime(VOCUS_PTR["Date_and_Time_(UTC)"], dayfirst = True, format = "mixed")

SO2_v1_data_1hour = lif.resample_data(SO2_v1_data, averaging = "60 min")
SO2_v1_data_1hour["Date_time"] = pd.to_datetime(SO2_v1_data_1hour["Date_time"])


fig, g = plt.subplots(3,1, sharex = True)
g[0].plot(VOCUS_PTR["Date_and_Time_(UTC)"],VOCUS_PTR["DMS_(ppt)"])
g[0].set_ylabel("DMS (ppt)", fontsize = 20)
g[1].plot(VOCUS_PTR["Date_and_Time_(UTC)"],VOCUS_PTR["DMSO2_(ppt)"])
g[1].set_ylabel("MeSH (ppt)", fontsize = 20)
g[2].plot(SO2_v1_data_1hour["Date_time"], SO2_v1_data_1hour["amb_SO2_ppt"])
g[2].set_ylabel("SO2 (ppt)", fontsize = 20)
