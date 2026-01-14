import lif_functions as lif
import pandas as pd


# --------------------- Variables to manually input --------------------------

molecule = 'NO'                                                                 # NO (even for multi-channel NOx) or SO2
channels = {'sig_A':'NO', 'sig_B':'NO2'}                                        # dict of all sig channels
start_times = {'sig_A':'30/05/2025 00:00', 'sig_B':'05/06/2025 00:00'}
cal_task = 5                                                                    # Task number associated with regular cals
BLC_cal_task = 1                                                                # Task number associated with BLC cals
R2_limit = 0.75                                                                 # Lower limit for R2 values of cal regression
BLC = True                                                                      # Boolean indicator of whether BLC present
plot = False                                                                    # diagnostics plots at various analysis stages
averaging = '1 min'                                                             # averaging for the final output file
filename = 'NOx_CARES_MaceHead_prelim_with_baseline_20260113'                                 # filename for the resampled MR output
cal_cylinder_conc = 5000                                                        # in ppb
pre_taskswitch = 300    # (at 10Hz = 30 secs)                                   # data points before task switch to ignore
post_taskswitch = 600   # (at 10Hz = 60 secs)                                   # data points after task switch to ignore
pre_peakfind = 20       # (at 10Hz = 2 secs)                                    # data points before ref_cts_diff drop to ignore
post_peakfind = 200     # (at 10Hz = 20 secs)                                   # data points after ref_cts_diff drop to ignore
ref_cts_diff_limit = 75000                                                      # lower limit ref_diff_cts_norm

data_dir = ('C:/Users/pp835/OneDrive - University of York/Documents/'
            'Data Analysis/CARES/Mace Head Full Data Analysis/Data'
            )
# ----------------------------------------------------------------------------

\
day_folders = lif.find_day_folders(
    data_dir
    )
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )
cts_data = lif.CARES_NO_ref_correction(
    cts_data
    )
cts_data_ref_norm = lif.ref_normalise(
    cts_data, channels=channels
    )
cts_data_flagged = lif.set_flags(
    cts_data_ref_norm, pre_taskswitch, post_taskswitch, pre_peakfind
    , post_peakfind, ref_cts_diff_limit
    )
cts_data_zeroed = lif.zero_correct_average(
    cts_data_flagged, channels=channels, plot=True
    )
lif.analyse_cals(
    cts_data_zeroed, data_dir, channels=channels, molecule=molecule
    , cal_cylinder_conc=cal_cylinder_conc, plot=plot, save_csv=True
    , cal_task=cal_task
    )
lif.analyse_BLC_cals(
    cts_data_zeroed, data_dir, plot=plot, save_csv=True
    , BLC_cal_task=BLC_cal_task
    )
MR_data = lif.apply_cals_average(
    data_dir, cts_data_zeroed, R2_limit, channels=channels, BLC=BLC
    , start_times=start_times
    )

mask = ((MR_data['Date_time'] < pd.to_datetime('06/06/2025 00:00', dayfirst=True)) |
        (MR_data['Date_time'] > pd.to_datetime('06/06/2025 12:00', dayfirst=True))
        )
MR_data = MR_data[mask]

MR_data_resampled = lif.resample_data(
    MR_data, averaging
    )
lif.save_to_csv(
    data_dir, filename, MR_data_resampled
    )
lif.CARES_NO_plot_data_old(data_dir, MR_data_resampled)


lif.CARES_NO_diurnal_raw_mean(
    data_dir, MR_data
    )
lif.CARES_NO_diurnal_raw_median(
    data_dir, MR_data
    )
lif.CARES_NO_diurnal_median_of_medians(
    data_dir, MR_data
    )
lif.CARES_NO_diurnal_mean_of_medians(
    data_dir, MR_data
    )
lif.CARES_NO_diurnal_median_of_means(
    data_dir, MR_data
    )
lif.CARES_NO_diurnal_mean_of_means(
    data_dir, MR_data
    )