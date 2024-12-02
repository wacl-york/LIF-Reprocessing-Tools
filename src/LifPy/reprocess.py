import datetime
import os
import sys
import numpy as np
import pandas as pd
import math
from itertools import islice
from datetime import datetime as dt
import shutil
import LifPy.lif_utils as lif_utils


def setup_LifEnv(path):

    LifPy_path = LifPy.__path__[0]

    LifPy.lif_utils.generate_folder(path + '\\lib', use_local_dir=False)

    file_arr = ['config.txt', 'cts_metadata.txt', 'misalligned_files.txt']

    for filename in file_arr:
        if not os.path.exists(os.path.join(path + '\\lib', filename)):
            shutil.copyfile(LifPy_path + '\\lib', filename, path + '\\lib\\' + filename)

    config_file = open(os.path.join(path + '\\lib\\config.txt'), 'w+')

    config_file.write('local_dir=' + path)

    config_file.close()

    if not os.path.exists(os.path.join(path + '\\reprocessing_examp.py')):
        shutil.copyfile(LifPy_path + '\\reprocessing_examp.py', os.path.join(path + '\\reprocessing_examp.py'))

    if not os.path.exists(os.path.join(path + '\\diag_plots_examp.py')):
        shutil.copyfile(LifPy_path + '\\diag_plots_examp.py', os.path.join(path + '\\diag_plots_examp.py'))

    for name in ['bin_data', 'HK_data', 'processed_data']:
        LifPy.lif_utils.generate_folder(path + '\\data\\%s' % name, use_local_dir=False)

    for name in ['diagnostics', 'calibrations']:
        LifPy.lif_utils.generate_folder(path + '\\figures\\%s' % name, use_local_dir=False)

    print('The LIF processing environment has been successfully setup!')


def reprocess_binary_data(log_start_datetime, HK_headers_dict, bin_file_path='data\\bin_data'
                          , HK_file_path='data\\HK_data', data_freq=10, skip_start=0, skip_end=0, ignore_first=False
                          , lag=0):

    config_path = (r'{}' + '\\lib\\config.txt').format(os.getcwd())
    config = {var.split('=')[0]: var.split('=')[1] for var in open(config_path, 'rt').read().split('\n')}

    HK_time_arr, HK_data = lif_utils.import_HK_data(config['local_dir'] + '\\' + HK_file_path)

    # The misalligned file process corrects for the fact that the seed LD mode may be offset in some counts files
    # file_shift returns a list of files, the parameters in those files which need to be shifted and in which direction

    file_shift = pd.read_csv(config['local_dir'] + '\\lib\\misalligned_files.txt', header=0, delimiter=',')

    # Produces a list of filenames based off the directory specified
    file_list = [f for f in os.listdir(bin_file_path) if os.path.isfile(os.path.join(bin_file_path, f))]

    # function to skip files
    try:
        file_list = file_list[skip_start: len(file_list) - skip_end]
    except Exception as err:
        if skip_end + skip_start >= len(file_list):
            print('Error - skip_start and skip_end are set to discard all binary files.')
        else:
            print(err)

    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers
    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
        , 'seed_LD_mode': 9}
    channel_count = 10

    # epoch time is the number of seconds between 01/01/1904 and the log_start_time
    try:
        epoch_time = (dt.strptime(log_start_datetime, '%d/%m/%Y %H:%M:%S') -
                      dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()
    except Exception as err:
        print('\n' + str(err))
        if 'does not match format' in str(err):
            print('\nTrying alternative timestamp formatter')
            try:
                epoch_time = (dt.strptime(log_start_datetime, '%H:%M:%S %d/%m/%Y')
                              - dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()
                print('Successfully coerced timestamp')
            except:
                print('\nUnable to convert log_start_datetime string, please check the input format. Exit - 01')
                sys.exit(1)

        if 'unconverted data remains:' in str(err):
            print('\nDeleting milliseconds from timestamp')
            try:
                epoch_time = (dt.strptime(log_start_datetime.replace(str(err).split(': ')[1], ''), '%d/%m/%Y %H:%M:%S')
                              - dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()
                print('Successfully coerced timestamp')
            except:
                print('\nUnable to convert log_start_datetime string, please check the input format. Exit - 01')
                sys.exit(1)


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

        binary_data_dict, lag_calc = lif_utils.interleave_binary_file(binary_data, channel_format, channel_count)

        if file == file_list[0]:
            if not math.isnan(lag_calc):
                # corrects for startup lag if found, the + 0.5 is so that int() always rounds up
                lag_sec = lag_calc / 1000
                HK_data['Time_s'] = HK_data['Time_s'] - int((lag_sec / 2) + 0.5)

                print('Shift HK data by %.0f seconds' % lag_sec)
            else:
                HK_data['Time_s'] = HK_data['Time_s'] - int((lag / 2) + 0.5)

        tot_steps = len(binary_data_dict['time_ms']) - 1

        # counts number of files in processed data folder and generates an index based on this number
        file_ind = str(len(os.listdir(config['local_dir'] + '\\data\\processed_data'))).zfill(2)

        binary_file = open(config['local_dir'] + '\\data\\processed_data\\%s_LIF_processed_data_%s.txt'
                           % (file.split(sep="_")[1][0: 8], file_ind), 'w+')

        met_add = '\nReprocessed on: ' + dt.strftime(dt.now(), '%Y/%m/%d %H:%M:%S') + \
                  '\nTime reference: (seconds since 1904-01-01 00:00:00)\nPermalink to reprocess code: \n'

        cts_met = '\n\n' + open(config['local_dir'] + '\\lib\\cts_metadata.txt').read() + met_add
        binary_file.write(str(cts_met.count('\n') - 1) + cts_met + '\n')

        binary_headers = 'mac_time_s,sig_on_cts,sig_off_cts,sig_cts_diff,ref_on_cts,ref_off_cts,ref_cts_diff,' \
                         'lsr_pwr_on_mW,lsr_pwr_off_mW,lsr_pwr_mW'

        for HK_ID in list(HK_headers_dict):
            try:
                HK_data[HK_headers_dict[HK_ID]]
            except Exception as err:
                print('\nThere is no HK data header called ' + str(err).replace("'", '') + ', so it will be discarded')
                del HK_headers_dict[HK_ID]

        HK_labels = ','.join(list(HK_headers_dict)) + '\n'

        all_headers = binary_headers + ',' + HK_labels
        binary_file.write(all_headers)

        nan_data = ','.join(np.full(len(all_headers.split(',')[1::]), str(-9999)))

        bin_time = [0]
        bin_time_arr = (binary_data_dict['time_ms'] / 1000) + epoch_time

        HK_start_ind = lif_utils.find_min_ind(bin_time_arr[0], HK_data['Time_s'])
        HK_end_ind = lif_utils.find_min_ind(bin_time_arr[-1], HK_data['Time_s'])

        if HK_start_ind == HK_end_ind:
            print('\nThe log_start_datetime is incorrect')
            sys.exit(1)

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

                    HK_ind = HK_start_ind + lif_utils.find_min_ind(curr_time, HK_data['Time_s'], HK_start_ind, HK_end_ind)

                    if not math.isnan(sig_cts):

                        bin_write = ','.join([str(curr_time), str(on_cts), str(off_cts), str(sig_cts), str(on_ref_cts)
                                             , str(off_ref_cts), str(ref_cts), str(on_lsr_pwr), str(off_lsr_pwr)
                                             , str(lsr_pwr)])
                        HK_write = ','.join([str(HK_data[HK_headers_dict[label]][HK_ind]) for label in
                                             list(HK_headers_dict)])

                        binary_file.write(bin_write + ',' + HK_write + '\n')

