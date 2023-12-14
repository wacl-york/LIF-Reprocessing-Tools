
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from datetime import datetime as dt

from gen_processes import generate_folder, find_min_ind


def import_HK_data(HK_file_path, skip_start=0, skip_end=0):
    # Uses pandas to import the LIF HK data files and concatenate them
    # files should be in a sub-directory which is provided to the call function as bin_file_path='subdirectory path'
    # skip_start and skip_end causes the function to not read the first or last files

    file_list = [f for f in listdir(HK_file_path) if isfile(join(HK_file_path, f))]

    HK_data = {}

    print('\nreading HK files:\n')

    for file in range(skip_start, len(file_list) - skip_end):
        print(file_list[file])
        file_data = pd.read_csv(HK_file_path + '/' + file_list[file], delimiter='\s+', header=0)

        for header in list(file_data):
            try:
                HK_data[header] = np.concatenate((HK_data[header], file_data[header]))
            except:
                HK_data[header] = np.array([])
                HK_data[header] = np.concatenate((HK_data[header], file_data[header]))

    return HK_data


def interleave_binary_file(binary_data, channel_format, channel_count):
    # Reads in and processes the binary file data, returning them as a human readable dictionary of numpy.array

    frames = np.array(binary_data)
    decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

    binary_data_dict = {}

    for dict_key in channel_format.items():
        # Populates the dictionary with keys corresponding to data channels

        binary_data_dict[dict_key[0]] = []

    for channel, channel_ID in channel_format.items():
        # Process to interleave the decimated data types. Not all data columns are decimated, as specified by the
        # channel ID

        if type(0) == type(channel_ID):
            binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], decimate_arr[channel_ID]))
        if type([]) == type(channel_ID):
            hi = decimate_arr[channel_ID[0]] * 65536
            lo = np.where(decimate_arr[channel_ID[1]] < 0
                          , decimate_arr[channel_ID[1]] + 65536, decimate_arr[channel_ID[1]])
            binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], lo + hi))

    # The remainder of this code converts the laser power from nW to mW, corrects the counts for the PMTs non-
    # lineararity, and normalises the counts to laser power in mW.

    binary_data_dict['laser_pwr_PT0'] = binary_data_dict['laser_pwr_PT0'] / 100000

    rep_rate = 200 * 1000  # Hz
    loop_period = 1 / 5  # seconds
    max_cts = rep_rate * loop_period

    binary_data_dict['sig_counts_norm'] = -np.log(1 - (binary_data_dict['sig_counts'] / max_cts)) * max_cts
    binary_data_dict['ref_counts_norm'] = -np.log(1 - (binary_data_dict['ref_counts'] / max_cts)) * max_cts

    binary_data_dict['sig_counts_norm'] = binary_data_dict['sig_counts'] / (binary_data_dict['laser_pwr_PT0'])
    binary_data_dict['ref_counts_norm'] = binary_data_dict['ref_counts'] / (binary_data_dict['laser_pwr_PT0'])

    binary_data_dict = check_FPGA_lag(binary_data_dict)

    return binary_data_dict


def check_FPGA_lag(binary_data):
    # A function to check the binary data for a lag in the data on the greater than a few milliseconds. This to prevent
    # the HK data array and binary data array from being offset to each other

    delta_bin = [t_1 - t_0 for t_1, t_0 in
                 zip(binary_data['time_ms'][1::], binary_data['time_ms'][0: len(binary_data['time_ms'])])]
    lag_ind = find_min_ind(20 * 1000, delta_bin)

    if delta_bin[lag_ind] > 100:
        binary_start_ind = lag_ind + 1

        bin_header_arr = list(binary_data)
        new_binary_data = {}

        for bin_header in bin_header_arr:
            new_binary_data[bin_header] = binary_data[bin_header][binary_start_ind::]

        return new_binary_data
    else:
        return binary_data


def reinterpolate_HK_file(HK_data, binary_data_dict, HK_headers, HK_freq, epoch_time):
    # Uses the binary and HK data frequencies (hard coded as of now) to reinterpolate the HK data to the same frequency
    # as the counts data. Therefore the two data sets are now at the same frequency and can be compared.

    ini_time = (binary_data_dict['time_ms'][0] / 1000) + epoch_time
    HK_start_ind = find_min_ind(ini_time, HK_data['Time_s'])

    bin_freq = 100
    HK_freq = 5

    processed_HK_dict = {}

    for header in HK_headers:
        processed_HK_dict[header] = np.array([])

    for i in range(int(len(binary_data_dict['time_ms']) / (bin_freq / HK_freq)) + 2):
        for header in HK_headers:
            processed_HK_dict[header] = np.concatenate((processed_HK_dict[header]
                                                        , np.array([HK_data[header][HK_start_ind + i]
                                                                       , HK_data[header][HK_start_ind + i]])))

    return processed_HK_dict


def corr_ref_cts(data):
    # A function to correct the signal counts by dividing through by the either the online - offline difference or
    # ratio in the reference counts

    ref_cts_diff = data['on_ref_cts_norm'] - data['off_ref_cts_norm']
    corr_cts = data['cts_norm'] / ref_cts_diff

    return corr_cts


def apply_mask(data, cal_ind_dict, zero_ind_dict):
    # Applys a mask of nan values to calibration periods.

    for cal_dict_key in cal_ind_dict:

        nan_start, nan_end = [min(cal_ind_dict[cal_dict_key]), max(cal_ind_dict[cal_dict_key])]

        nan_arr = np.empty(nan_end - nan_start + 1)
        nan_arr[:] = np.nan
        data.loc[nan_start: nan_end, 'cts_norm'] = nan_arr

    for zero_dict_key in zero_ind_dict:

        nan_start, nan_end = [min(zero_ind_dict[zero_dict_key]), max(zero_ind_dict[zero_dict_key])]

        nan_arr = np.empty(nan_end - nan_start + 1)
        nan_arr[:] = np.nan
        data.loc[nan_start: nan_end, 'cts_norm'] = nan_arr


def write_txt(data, delimiter, folder_name):
    # A function to write data to a specific folder, using a specific delimiter. The data provided must be in the
    # format of a dictionary of lists or arrays. The first key in that dictionary must correspond to the time stamp.

    generate_folder(folder_name)

    time_str = dt.strftime(dt.fromtimestamp(data[list(data)[0]][0] - 2082844800), '%Y%m%d')
    mr_file = open('%s/%s_LIF_mr_data.txt' % (folder_name, time_str), 'w+')
    header = str(list(data)).replace(']', '').replace('[', '').replace(' ', '').replace("'", '') + '\n'
    mr_file.write(header)

    for i in range(len(data[list(data)[0]])):
        new_time_str = dt.strftime(dt.fromtimestamp(data[list(data)[0]][i] - 2082844800), '%Y%m%d')

        if new_time_str != time_str:
            mr_file.close()
            time_str = new_time_str

            mr_file = open('%s/%s_LIF_mr_data.txt' % (folder_name, time_str), 'w+')
            header = str(list(data)).replace(']', '').replace('[', '').replace(' ', '').replace("'", '') + '\n'
            mr_file.write(header)

        for key in list(data):
            if data[key][i] != -9999:
                mr_file.write(str(data[key][i]))
            else:
                mr_file.write(str(np.nan))
            if list(data).index(key) == len(list(data)) - 1:
                mr_file.write('\n')
            else:
                mr_file.write(delimiter)

    mr_file.close()

