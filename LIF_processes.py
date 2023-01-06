
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from datetime import datetime as dt
import math


def import_HK_data(bin_file_path, skip_start=0, skip_end=0):
    # Uses pandas to import the LIF HK data files and concatenate them
    # files should be in a sub-directory which is provided to the call function as bin_file_path='subdirectory path'
    # skip_start and skip_end causes the function to not read the first or last files

    file_list = [f for f in listdir(bin_file_path) if isfile(join(bin_file_path, f))]

    file_data = []

    print('\nReading file:\n')
    try:
        for file in range(skip_start, len(file_list) - skip_end):
            print(file)
            file_data.append(pd.read_csv(bin_file_path + '/' + file_list[file], delimiter='\s+', header=0))

        HK_data = {}

        for i in range(len(list(file_data[0]))):
            data_lines = []
            for k in range(len(file_data)):
                data_lines.append(file_data[k][list(file_data[0])[i]])
            HK_data[list(file_data[0])[i]] = np.concatenate((data_lines))

        return HK_data, list(HK_data)

    except:
        print('Files not found or unreadable, check the file path, skip_start and skip_end')
        return 'Empty data array', 'Empty header array'


def reprocess_binary_data(bin_file_path, log_start_datetime, sensitivity=1, bckgrnd=0, data_freq=10, skip_start=0
                          , skip_end=0, dtype='>i2'):
    epoch_time = (dt.strptime(log_start_datetime, '%d/%m/%Y %H:%M:%S') - dt.strptime('01/01/1904',
                                                                                     '%d/%m/%Y')).total_seconds()
    # epoch time is the number of seconds between 01/01/1904 and the log_start_time

    file_list = [f for f in listdir(bin_file_path) if isfile(join(bin_file_path, f))]
    # Produces a list of filenames based off the directory specified

    filt_file_list = file_list[skip_start: len(file_list) - skip_end]  # function to skip files

    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
                      , 'seed_LD_mode': 9}
    channel_count = 10
    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers

    for file in filt_file_list:

        print('\nReprocessing file %s' % file)
        # to let you know which file is being looked at

        data = np.fromfile(bin_file_path + '/' + file, dtype=dtype)

        frames = np.array(data)
        decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

        binary_data_dict = {}

        for dict_key in channel_format.items():
            binary_data_dict[dict_key[0]] = []

        for channel, channel_ID in channel_format.items():
            if type(0) == type(channel_ID):
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], decimate_arr[channel_ID]))
            if type([]) == type(channel_ID):
                hi = decimate_arr[channel_ID[0]] * 65536
                lo = np.where(decimate_arr[channel_ID[1]] < 0
                              , decimate_arr[channel_ID[1]] + 65536, decimate_arr[channel_ID[1]])
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], lo + hi))

        binary_data_dict['sig_counts_norm'] = binary_data_dict['sig_counts'] / (binary_data_dict['laser_pwr_PT0'] / 100000)

        reprocessed_data_dict = {}

        keys = ['Time_ms', 'SO2_mr', 'On_cts_time_ms', 'On_cts', 'On_cts_norm', 'On_lsr_power_V'
                , 'Off_cts_time_ms', 'Off_cts', 'Off_cts_norm', 'Off_lsr_power_V', 'Cts_diff']

        for key in keys:
            reprocessed_data_dict[key] = []

        group_avg = int((1 / data_freq) * 10)
        skip_set = (100 * group_avg) - 100  # skip_set = 100 yields 5 Hz data

        dir_path = os.path.dirname(__file__)

        path = r'{}/processed data'.format(dir_path)

        try:
            os.makedirs(path)
        except OSError:
            pass

        binary_file = open('processed data/%s_SO2_processed_data_%.0f.txt'
                           % (file.split(sep="_")[1].split(sep=" ")[0], file_list.index(file)), 'w+')
        binary_file.write('time_ms,SO2_pptv,on_cts_norm,off_cts_norm,off_lsr_pwr_V,off_lsr_pwr_V\n')

        tot_steps = len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)

        SO2_mr = []
        cts_diff = []
        off_counts_norm = []
        off_counts = []
        laser_pwr_off = []
        on_counts_norm = []
        on_counts = []
        laser_pwr_on = []
        bin_time = [0]
        time_on = []
        time_off = []
        mac_time = []

        for i in range(len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)):
            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100), end='')

            curr_time = binary_data_dict['time_ms'][i] + (epoch_time * 1000)

            if binary_data_dict['seed_LD_mode'][i] == 5:
                time_off.append(curr_time)
                off_counts_norm.append(binary_data_dict['sig_counts_norm'][i])
                off_counts.append(binary_data_dict['sig_counts'][i])
                laser_pwr_off.append(binary_data_dict['laser_pwr_PT0'][i])

            if binary_data_dict['seed_LD_mode'][i] == 6:
                time_on.append(curr_time)
                on_counts_norm.append(binary_data_dict['sig_counts_norm'][i])
                on_counts.append(binary_data_dict['sig_counts'][i])
                laser_pwr_on.append(binary_data_dict['laser_pwr_PT0'][i])

            if binary_data_dict['seed_LD_mode'][i] == 6 and binary_data_dict['seed_LD_mode'][i + 1] == 5:
                if binary_data_dict['time_ms'][i] - bin_time[-1] != skip_set:

                    bin_time.append(binary_data_dict['time_ms'][i])
                    mac_time.append(curr_time)

                    on_cts = []
                    on_lsr_pwr = []
                    for j in range(group_avg):
                        on_cts.append(binary_data_dict['sig_counts_norm'][i - 7 - (j * 10): i + 1 - (j * 10)])
                        on_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i - 7 - (j * 10): i + 1 - (j * 10)])

                    off_cts = []
                    off_lsr_pwr = []
                    for j in range(group_avg):
                        off_cts.append(binary_data_dict['sig_counts_norm'][i + 1 - (j * 10): i + 3 - (j * 10)])
                        off_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i + 1 - (j * 10): i + 3 - (j * 10)])

                    set_len = (group_avg * len(on_cts[0])) + (group_avg * len(off_cts[0]))

                    mix_r = ((((np.mean(on_cts) - np.mean(off_cts)) * set_len) - (bckgrnd / data_freq))
                             / (sensitivity * (((1 / data_freq) * 1000) / 1000)))
                    cts = (np.mean(on_cts) - np.mean(off_cts)) * set_len * data_freq

                    SO2_mr.append(mix_r)
                    cts_diff.append(cts)

                    if not math.isnan(mix_r):

                        binary_file.write(str(curr_time) + ',')
                        binary_file.write(str(mix_r) + ',')
                        binary_file.write(str(np.mean(off_cts)) + ',')
                        binary_file.write(str(np.mean(on_cts)) + ',')
                        binary_file.write(str(np.mean(off_lsr_pwr)) + ',')
                        binary_file.write(str(np.mean(on_lsr_pwr)) + '\n')

