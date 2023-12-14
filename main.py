
import os
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import islice
from datetime import datetime as dt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gen_processes import generate_folder, find_min_ind, search_tsk
from LIF_processes import import_HK_data, reinterpolate_HK_file, interleave_binary_file, write_txt, corr_ref_cts, apply_mask


def linear(fit, x):
    return fit[0] * x + fit[1]


def reprocess_binary_data(bin_file_path, HK_file_path, log_start_datetime, data_freq=10, skip_start=0, skip_end=0
                          , diag_plots=[], ignore_first=True):
    restart_t = dt.strptime('13:24:00', '%H:%M:%S')

    epoch_time_arr = [(dt.strptime(t, '%d/%m/%Y %H:%M:%S') - dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds() for
                      t in log_start_datetime]

    time_shift = [(dt.strptime(t.split(sep=' ')[1], '%H:%M:%S') - restart_t).total_seconds() for t in
                  log_start_datetime[1::]]
    time_shift = [0] + time_shift
    time_shift = [t_1 + t_0 for t_1, t_0 in zip(time_shift[1::], time_shift[0: len(time_shift)])]

    HK_data = import_HK_data(HK_file_path)
    HK_time_arr = [dt.fromtimestamp(t) if t + 2082844800 != -9999 else np.nan for t in
                   HK_data['Time_s'][1::] - 2082844800]

    if diag_plots == 'all':

        for header in list(HK_data):
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=HK_time_arr, y=HK_data[header]
                , mode='lines'
                , name='%s' % header
            ))
            fig.write_html('figures/diagnostics/%s.html' % header)
            fig.show()

    else:

        for header in diag_plots:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=HK_time_arr, y=HK_data[header]
                , mode='lines'
                , name='%s' % header
            ))
            fig.write_html('figures/diagnostics/%s.html' % header)
            fig.show()

    misalligned_file = open('misalligned_files.txt', 'w+')

    misalligned_file.write('name,shift\n')
    misalligned_file.close()

    await_file_names = input('\nAdd file names to misalligned_files.txt, type y and press enter when complete\n')

    file_shift = pd.read_csv('misalligned_files.txt', header=0, delimiter=',')

    HK_freq = 1 / np.mean(
        [t_2 - t_1 for t_1, t_2 in zip(HK_data['Time_s'][0: len(HK_data['Time_s']) - 1], HK_data['Time_s'][1::])])

    # epoch time is the number of seconds between 01/01/1904 and the log_start_time

    file_list = [f for f in os.listdir(bin_file_path) if os.path.isfile(os.path.join(bin_file_path, f))]
    # Produces a list of filenames based off the directory specified

    filt_file_list = file_list[skip_start: len(file_list) - skip_end]  # function to skip files
    print(filt_file_list)

    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
        , 'seed_LD_mode': 9}
    channel_count = 10
    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers

    for file in filt_file_list:

        print('\nReprocessing file %s' % file)
        # to let you know which file is being looked at

        if file.split(sep='.')[0] in list(file_shift['name']):
            shift = file_shift['shift'][list(file_shift['name']).index(file.split(sep='.')[0])]
            print('Shifting seed LD mode by %.0f' % shift)
        else:
            shift = 0

        binary_data = np.fromfile(bin_file_path + '/' + file, dtype='>i2')

        binary_data_dict = interleave_binary_file(binary_data, channel_format, channel_count)

        HK_headers = ['Time_s', 'Task', 'Cal_SO2_MFC_Read', 'Cell_Flow', 'Seed_Laser_T_Read']
        processed_HK_dict = reinterpolate_HK_file(HK_data, binary_data_dict, HK_headers, HK_freq, epoch_time_arr[0])

        group_avg = int((1 / data_freq) * 10)
        skip_set = (100 * group_avg) - 100  # skip_set = 100 yields 5 Hz data
        tot_steps = len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)

        generate_folder('processed data')

        file_ind = str(file_list.index(file)) if len(str(file_list.index(file))) > 1 else '0' + str(
            file_list.index(file))

        binary_file = open('processed data/%s_LIF_processed_data_%s.txt'
                           % (file.split(sep="_")[1][0: 8], file_ind), 'w+')
        binary_file.write(
            'time_s,on_cts_norm,off_cts_norm,cts_norm,off_lsr_pwr_V,on_lsr_pwr_V,off_ref_cts_norm,on_ref_cts_norm,task'
            ',cal_no_mfc,cell_flow,seed_laser_t\n')

        bin_time = [0]
        sig_cts_arr = []

        iter_range = iter(range(tot_steps))

        for i in iter_range:

            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100), end='')

            curr_time = (binary_data_dict['time_ms'][i] / 1000) + epoch_time_arr[0]

            delta_arr = [curr_time - ref_t for ref_t in epoch_time_arr]

            for g in range(len(delta_arr)):
                if delta_arr[g] > 0:
                    curr_time = curr_time + time_shift[g]

            if binary_data_dict['seed_LD_mode'][i + shift] != 1 and binary_data_dict['seed_LD_mode'][
                i + 1 + shift] == 1:
                for j in range(tot_steps - i):
                    if binary_data_dict['seed_LD_mode'][i + j + shift] == 1 and binary_data_dict['seed_LD_mode'][
                        i + j + 1 + shift] != 1:

                        time_fill = list(
                            np.arange(binary_data_dict['time_ms'][i], binary_data_dict['time_ms'][i + j] + 1
                                      , 100))
                        bin_time = bin_time + time_fill

                        nan_arr = np.empty(int((j / 10) + 1))
                        nan_arr[:] = np.nan
                        sig_cts_arr = sig_cts_arr + list(nan_arr)

                        for x in range(int(j / 10) + 1):
                            binary_file.write(str(curr_time + (1 / data_freq) * x) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(3) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + ',')
                            binary_file.write(str(-9999) + '\n')
                        next(islice(iter_range, j, i), None)
                        break

            if binary_data_dict['seed_LD_mode'][i + shift] == 6 and binary_data_dict['seed_LD_mode'][
                i + 1 + shift] == 5:
                if binary_data_dict['time_ms'][i] - bin_time[-1] != skip_set:

                    bin_time.append(binary_data_dict['time_ms'][i])

                    if ignore_first:
                        offset = 6
                    else:
                        offset = 7

                    on_cts = []
                    on_lsr_pwr = []
                    on_ref_cts = []
                    for j in range(group_avg):
                        on_cts.append(binary_data_dict['sig_counts_norm'][i - offset - (j * 10): i + 1 - (j * 10)])
                        on_ref_cts.append(binary_data_dict['ref_counts_norm'][i - offset - (j * 10): i + 1 - (j * 10)])
                        on_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i - offset - (j * 10): i + 1 - (j * 10)])

                    off_cts = []
                    off_lsr_pwr = []
                    off_ref_cts = []
                    for j in range(group_avg):
                        off_cts.append(binary_data_dict['sig_counts_norm'][i + 1 - (j * 10): i + 3 - (j * 10)])
                        off_ref_cts.append(binary_data_dict['ref_counts_norm'][i + 1 - (j * 10): i + 3 - (j * 10)])
                        off_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i + 1 - (j * 10): i + 3 - (j * 10)])

                    set_len = (group_avg * len(on_cts[0])) + (group_avg * len(off_cts[0]))

                    sig_cts_diff = np.mean(on_cts) - np.mean(off_cts)
                    sig_cts = sig_cts_diff * set_len * data_freq
                    sig_cts_arr.append(sig_cts)

                    if not math.isnan(sig_cts):
                        binary_file.write(str(curr_time) + ',')
                        binary_file.write(str(np.mean(on_cts)) + ',')
                        binary_file.write(str(np.mean(off_cts)) + ',')
                        binary_file.write(str(np.mean(sig_cts)) + ',')
                        binary_file.write(str(np.mean(off_lsr_pwr)) + ',')
                        binary_file.write(str(np.mean(on_lsr_pwr)) + ',')
                        binary_file.write(str(np.mean(off_ref_cts)) + ',')
                        binary_file.write(str(np.mean(on_ref_cts)) + ',')
                        binary_file.write(str(processed_HK_dict['Task'][int(i / 10)]) + ',')
                        binary_file.write(str(processed_HK_dict['Cal_SO2_MFC_Read'][int(i / 10)]) + ',')
                        binary_file.write(str(processed_HK_dict['Cell_Flow'][int(i / 10)]) + ',')
                        binary_file.write(str(processed_HK_dict['Seed_Laser_T_Read'][int(i / 10)]) + '\n')


def process_cals(data, skip_cal, cal_ind_dict, zero_ind_dict, cyl_ppmv):

    cal_time, cal_sens, cal_int, zero_time, zero_arr = ([], [], [], [], [])

    generate_folder('figures/calibrations')

    for dict_key in cal_ind_dict:
        cal_ind_arr = cal_ind_dict[dict_key]

        tot_steps = round((cal_ind_arr[1] - cal_ind_arr[0]) / (10 * skip_cal['tot_sec']))
        step_len = round((cal_ind_arr[1] - cal_ind_arr[0]) / tot_steps)

        mean_cts_arr = [np.mean(data['cts_norm'][cal_ind_arr[0] + (step_len * step) + skip_cal['start']
                                                 : cal_ind_arr[0] + (step_len * (step + 1)) - skip_cal['end']])
                        for step in np.arange(0, tot_steps, 1)]
        NO_MFC_arr = [np.mean(data['cal_no_mfc'][cal_ind_arr[0] + (step_len * step) + skip_cal['start']
                                               : cal_ind_arr[0] + (step_len * (step + 1)) - skip_cal['end']])
                      for step in np.arange(0, tot_steps, 1)]
        tot_flow_arr = [np.mean(data['cell_flow'][cal_ind_arr[0] + (step_len * step) + skip_cal['start']
                                                 : cal_ind_arr[0] + (step_len * (step + 1)) - skip_cal['end']])
                        for step in np.arange(0, tot_steps, 1)]

        NO_mr_arr = list((cyl_ppmv * 1000 * np.array(NO_MFC_arr)) / (np.array(tot_flow_arr) * 1.061))

        near_zero = find_min_ind(cal_ind_arr[0], np.array([val[1] for val in zero_ind_dict.values()]))

        time_arr = [data['time_s'][cal_ind_arr[0] + (step_len * step) + skip_cal['start']
                                    : cal_ind_arr[0] + (step_len * (step + 1)) - skip_cal['end']]
                   for step in np.arange(0, tot_steps, 1)]
        cts_arr = [data['cts_norm'][cal_ind_arr[0] + (step_len * step) + skip_cal['start']
                                    : cal_ind_arr[0] + (step_len * (step + 1)) - skip_cal['end']]
                   for step in np.arange(0, tot_steps, 1)]

        if abs(zero_ind_dict[near_zero][1] - cal_ind_arr[0]) < 6000:
            NO_mr_arr.append(0)
            zero_cts_arr = data['cts_norm'][zero_ind_dict[near_zero][0] + (2 * skip_cal['start'])
                                            : zero_ind_dict[near_zero][1] - (2 * skip_cal['end'])]
            zero_arr.append(list(zero_cts_arr))
            mean_cts_arr.append(np.mean(zero_cts_arr))
            zero_time.append(data['time_s'][zero_ind_dict[near_zero][0]])

        for time, cts in zip(time_arr, cts_arr):
            plt.plot(time, cts)

        plt.savefig('figures/calibrations/timeseries_%s.png'
                    % (dt.fromtimestamp(data['time_s'][cal_ind_arr[0]] - 2082844800)).strftime('%Y_%m_%d_%H'))
        plt.show()

        cal_fit, cov = np.polyfit(NO_mr_arr, mean_cts_arr, 1, cov=True)

        y_pred = linear(cal_fit, np.array(NO_mr_arr))
        r2 = r2_score(mean_cts_arr, y_pred)

        plt.scatter(NO_mr_arr, mean_cts_arr)
        x_arr = np.arange(0, np.max(NO_mr_arr) + 100, 100)
        plt.plot(x_arr, cal_fit[0] * x_arr + cal_fit[1], color='lightsteelblue')

        plt.xlabel('NO amount fraction / pptv')
        plt.ylabel('counts s$^{-1}$ mW$^{-1}$')
        plt.annotate('m = %.2f counts s$^{-1}$ mW$^{-1}$ pptv$^{-1}$\nc = %.0f counts s$^{-1}$ mW$^{-1}$ \nR$^2$ = %.2f'
                    % (cal_fit[0], cal_fit[1], r2), xycoords='axes fraction', xy=(0.025, 0.8))

        plt.savefig('figures/calibrations/fit_%s.png'
                    % (dt.fromtimestamp(data['time_s'][cal_ind_arr[0]] - 2082844800)).strftime('%Y_%m_%d_%H'))
        plt.show()

        cal_input = input('Good calibration (y/n)?')

        if cal_input == 'y':
            cal_time.append(data['time_s'][cal_ind_arr[0]])
            cal_sens.append(cal_fit[0])
            cal_int.append(cal_fit[1])

    cal_results = {
        'time_s': cal_time
        , 'sensitivity': cal_sens
        , 'intercept': cal_int
    }

    zero_results = {
        'time_s': zero_time
        , 'zero arr': zero_arr
    }

    plt.errorbar([dt.fromtimestamp(t) for t in zero_results['time_s']]
                 , [np.mean(zero) for zero in zero_arr] / np.mean(cal_results['sensitivity'])
                 , yerr= [np.std(zero) for zero in zero_arr] / np.mean(cal_results['sensitivity'])
                 , ls=' ', marker='o', capsize=3)
    plt.xlabel('Time / UTC')
    plt.ylabel('NO zero mixing ratio / pptv')
    plt.yscale('log')

    plt.savefig('figures/zero_t_series.jpeg')
    plt.show()

    zero_mr = [item for sublist in zero_arr for item in sublist] / np.mean(cal_results['sensitivity'])

    mean, stdev, shapiro_wilk = [np.mean(zero_mr), np.std(zero_mr), stats.shapiro(zero_mr)]

    hist_data = plt.hist(zero_mr, color='lightsteelblue', bins=20, density=True)
    bin_delta = [abs(val - next_val) for val, next_val in zip(hist_data[1][0: len(hist_data[1]) - 1], hist_data[1][1::])]

    gauss_fit_x = np.arange(plt.xlim()[0], plt.xlim()[1], 1)
    gauss_fit_y = stats.norm.pdf(gauss_fit_x, mean, stdev)
    plt.plot(gauss_fit_x, gauss_fit_y, color='steelblue')

    plt.ylabel('Normalised Density / AU')
    plt.xlabel('NO$_2$ zero mixing ratio / pptv')

    plt.annotate('mean zero \u00B1 1\u03C3 = %.0f \u00B1 %.0f pptv\nShapiro-Wilk p-value = %.4f\nbin size = %.0f pptv'
                 % (mean, stdev, 1, np.nanmean(bin_delta))
                 , xycoords='axes fraction', xy=(0.02, 0.86))

    plt.savefig('figures/zero_dist.jpeg')
    plt.show()

    return cal_results


def calc_mr(data, cal_results, mode='constant'):

    if mode == 'constant':
        mean_sens = np.nanmean(cal_results['sensitivity'])

        print('Applying a constant sensitivity of %.2f' % mean_sens)

        sens_arr = np.empty(len(data['time_s']))
        sens_arr[:] = mean_sens

        writer_data = {
            'time_s': data['time_s']
            , 'mix_ratio': data['cts_norm'] / mean_sens
            , 'sensitivity': sens_arr
        }

    return writer_data


def process_mix_ratio(cal_application, skip_cal, cyl_ppmv, ref_corr=True, diag_plots=False):
    file_list = [f for f in os.listdir('processed data') if os.path.isfile(os.path.join('processed data', f))]
    cts_data = pd.DataFrame()
    print('Loading processed data:\n')

    for file in file_list:
        print(file)
        cts_data = pd.concat([
            cts_data
            , pd.read_csv('processed data/' + file)
        ], ignore_index=True)

    if ref_corr:
        cts_data['cts_norm'] = corr_ref_cts(cts_data)

    cal_ind_dict = search_tsk(cts_data, 5)
    zero_ind_dict = search_tsk(cts_data, 4)

    print('\nFound %.0f calibrations and %.0f zero measurements\n' % (len(list(cal_ind_dict))
                                                                      , len(list(zero_ind_dict))))

    cal_results = process_cals(cts_data, skip_cal, cal_ind_dict, zero_ind_dict, cyl_ppmv)

    if diag_plots:
        generate_folder('figures/diagnostics')

    apply_mask(cts_data, cal_ind_dict, zero_ind_dict)

    write_results = calc_mr(cts_data, cal_results)

    write_txt(write_results, ',', 'mix ratio data')

