
import os
import numpy as np
import pandas as pd
import math
from itertools import islice
from datetime import datetime as dt
import LifPy.utils as utils
import LifPy.lif_utils as lif_utils


def reprocess_binary_data(log_start_datetime, bin_file_path='data\\bin_data', HK_file_path='data\\HK_data', data_freq=10, skip_start=0, skip_end=0
                          , ignore_first=False, gen_diag_plots=True):

    config_path = (r'{}' + '\\bin\\config.txt').format(os.getcwd())
    print(config_path)
    config = {var.split('=')[0]: var.split('=')[1] for var in open(config_path, 'rt').read().split('\n')}

    HK_time_arr, HK_data = lif_utils.import_HK_data(HK_file_path)

    html_prefs = lif_utils.load_html_prefs()

    if gen_diag_plots == True:

        for plot_key in list(html_prefs):

            if '[' and ']' in html_prefs[plot_key]['y_header']:
                y_arr = {header: HK_data[header] for header in
                         html_prefs[plot_key]['y_header'].replace('[', '').replace(']', '').split(sep=',')}

            else:
                y_arr = HK_data[html_prefs[plot_key]['y_header']]

            utils.gen_HTML_plots(HK_data[html_prefs[plot_key]['x_header']]
                                 , y_arr, html_prefs[plot_key]['name'])

    # The misalligned file process corrects for the fact that the seed LD mode may be offset in some counts files
    if not os.path.exists('bin\\misalligned_files.txt'):

        misalligned_file = open('bin\\misalligned_files.txt', 'w+')
        misalligned_file.write('name,shift\n')
        misalligned_file.close()

    await_file_names = input('\nAdd file names to misalligned_files.txt, type y and press enter when complete\n')

    # file_shift returns a list of files, the parameters in those files which need to be shifted and in which direction
    file_shift = pd.read_csv('bin\\misalligned_files.txt', header=0, delimiter=',')

    # Produces a list of filenames based off the directory specified
    file_list = [f for f in os.listdir(bin_file_path) if os.path.isfile(os.path.join(bin_file_path, f))]

    # function to skip files
    file_list = file_list[skip_start: len(file_list) - skip_end]

    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers
    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
        , 'seed_LD_mode': 9}
    channel_count = 10

    # epoch time is the number of seconds between 01/01/1904 and the log_start_time
    epoch_time = (dt.strptime(log_start_datetime, '%d/%m/%Y %H:%M:%S') -
                  dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()

    for file in file_list:

        print('\nReprocessing file %s' % file)

        # returns dict containing neccassary shifts to binary data
        shift_dict = lif_utils.gen_shift_dict(file, file_shift)

        shf_sig, shf_ref, shf_LD_mode, shf_lsr = [
            shift_dict['sig_counts_norm']
            , shift_dict['ref_counts_norm']
            , shift_dict['seed_LD_mode']
            , shift_dict['laser_pwr_PT0']
        ]

        binary_data = np.fromfile(bin_file_path + '/' + file, dtype='>i2')

        binary_data_dict, lag = lif_utils.interleave_binary_file(binary_data, channel_format, channel_count)

        if file == file_list[0]:
            if not math.isnan(lag):
                # corrects for startup lag if found, the + 0.5 is so that int() always rounds up
                lag_sec = lag / 1000
                HK_data['Time_s'] = HK_data['Time_s'] - int((lag_sec / 2) + 0.5)

                print('Shift HK data by %.0f seconds' % lag_sec)
            else:
                lag = int(input('Enter expected HK, Binary data lag time in seconds:\n'))
                HK_data['Time_s'] = HK_data['Time_s'] - int((lag / 2) + 0.5)

        tot_steps = len(binary_data_dict['time_ms']) - 1

        # counts number of files in processed data folder and generates an index based on this number
        file_ind = str(len(os.listdir('data\\processed_data'))).zfill(2)

        binary_file = open('data\\processed_data\\%s_LIF_processed_data_%s.txt'
                           % (file.split(sep="_")[1][0: 8], file_ind), 'w+')

        binary_headers = 'mac_time_s,sig_on_cts,sig_off_cts,sig_cts_diff,ref_on_cts,ref_off_cts,ref_cts_diff,' \
                         'lsr_pwr_on_mW,lsr_pwr_off_mW,lsr_pwr_mW'

        HK_header_start = pd.read_csv('bin\\cts_file_headers.txt', header=None, delim_whitespace=True
                                      , nrows=1)[1][0]
        HK_headers_file = pd.read_csv('bin\\cts_file_headers.txt', skiprows=HK_header_start
                                      , header=None, delimiter='=', skipinitialspace=True)
        HK_headers_dict = {label.replace(' ', ''): HK_ID for label, HK_ID in zip(HK_headers_file[0], HK_headers_file[1])}
        HK_labels = ','.join(list(HK_headers_dict)) + '\n'

        all_headers = binary_headers + ',' + HK_labels
        binary_file.write(all_headers)

        nan_data = ','.join(np.full(len(all_headers.split(',')[1::]), str(-9999)))

        bin_time = [0]
        bin_time_arr = (binary_data_dict['time_ms'] / 1000) + epoch_time

        HK_start_ind = utils.find_min_ind(bin_time_arr[0], HK_data['Time_s'])
        HK_end_ind = utils.find_min_ind(bin_time_arr[-1], HK_data['Time_s'])
        # The start and end ind is the slice of HK data which overlaps with the binary data

        iter_range = iter(range(tot_steps))

        for i in iter_range:

            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100), end='')

            curr_time = bin_time_arr[i]

            if binary_data_dict['seed_LD_mode'][i + shf_LD_mode] != 1 and \
                    binary_data_dict['seed_LD_mode'][i + 1 + shf_LD_mode] == 1:
                for j in range(tot_steps - i):
                    if binary_data_dict['seed_LD_mode'][i + j + shf_LD_mode] == 1 and \
                            binary_data_dict['seed_LD_mode'][i + j + 1 + shf_LD_mode] != 1:

                        time_fill = list(
                            np.arange(binary_data_dict['time_ms'][i], binary_data_dict['time_ms'][i + j] + 1
                                      , 100))
                        bin_time = bin_time + time_fill

                        for k in range(int(j / data_freq) + 1):
                            binary_file.write(str(curr_time + (1 / data_freq) * k) + ',' + nan_data + '\n')

                        next(islice(iter_range, j, i), None)
                        break

            if binary_data_dict['seed_LD_mode'][i + shf_LD_mode] == 6 and \
                    binary_data_dict['seed_LD_mode'][i + 1 + shf_LD_mode] == 5:
                if binary_data_dict['time_ms'][i] - bin_time[-1] != 0:

                    bin_time.append(binary_data_dict['time_ms'][i])

                    if ignore_first:
                        offset = 6 # This removes the first online point, which can sometimes by bias low
                    else:
                        offset = 7

                    on_cts = np.mean(binary_data_dict['sig_counts_norm'][i - offset + shf_sig: i + 1 + shf_sig])
                    on_ref_cts = np.mean(binary_data_dict['ref_counts_norm'][i - offset + shf_ref: i + 1 + shf_ref])
                    on_lsr_pwr = np.mean(binary_data_dict['laser_pwr_PT0'][i - offset + shf_lsr: i + 1 + shf_lsr])

                    off_cts = np.mean(binary_data_dict['sig_counts_norm'][i + 1 + shf_sig: i + 3 + shf_sig])
                    off_ref_cts = np.mean(binary_data_dict['ref_counts_norm'][i + 1 + shf_ref: i + 3 + shf_ref])
                    off_lsr_pwr = np.mean(binary_data_dict['laser_pwr_PT0'][i + 1 + shf_lsr: i + 3 + shf_lsr])

                    sig_cts = (on_cts - off_cts) * 10 * data_freq
                    ref_cts = (on_ref_cts - off_ref_cts) * 10 * data_freq
                    lsr_pwr = np.mean([on_lsr_pwr, off_lsr_pwr])

                    HK_ind = HK_start_ind + utils.find_min_ind(curr_time, HK_data['Time_s'], HK_start_ind, HK_end_ind)

                    if not math.isnan(sig_cts):

                        bin_write = ','.join([str(curr_time), str(on_cts), str(off_cts), str(sig_cts), str(on_ref_cts)
                                             , str(off_ref_cts), str(ref_cts), str(on_lsr_pwr), str(off_lsr_pwr)
                                             , str(lsr_pwr)])
                        HK_write = ','.join([str(HK_data[HK_headers_dict[label]][HK_ind]) for label in
                                             list(HK_headers_dict)])

                        binary_file.write(bin_write + ',' + HK_write + '\n')

