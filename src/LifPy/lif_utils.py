
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objects as go


def generate_folder(path, use_local_dir=True):

    if use_local_dir == True:
        dir_path = os.path.dirname(__file__)
        path = (r'{}/' + path).format(dir_path)

    try:
        os.makedirs(path)

    except OSError:
        pass


def find_min_ind(target, array, start=0, end='full'):

    if end == 'full':
        array = array[start::]
    else:
        array = array[start: end]

    diff_arr = list(abs(np.array(array) - target))

    return diff_arr.index(np.min(diff_arr))


def gen_HTML_plots(x_arr, y_dict, name):

    fig = go.Figure()

    for y_key in list(y_dict):

        fig.add_trace(go.Scatter(
            x=x_arr, y=y_dict[y_key]
            , mode='lines'
            , name=y_key
        ))

    fig.write_html('figures/diagnostics/%s.html' % name)
    fig.show()


def load_html_prefs():
    html_file_prefs = open('bin/html_plots_prefs.txt', 'rt').read().split('\n')

    html_file_prefs = ''.join(html_file_prefs[int(html_file_prefs[0][2])::]).replace(' ', '').split('}')[:-1]

    html_dict = {int(ID.split('={')[0]): {val.split('=')[0]: val.split('=')[1] for val in
                                          ID.split('={')[1].split('\t')} for ID in html_file_prefs}

    return html_dict


def import_HK_data(HK_file_path, skip_start=0, skip_end=0):
    # Uses pandas to import the LIF HK data files and concatenate them
    # files should be in a sub-directory which is provided to the call function as bin_file_path='subdirectory path'
    # skip_start and skip_end causes the function to not read the first or last files

    file_list = [f for f in os.listdir(HK_file_path) if os.path.isfile(os.path.join(HK_file_path, f))]

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

    HK_time_arr = [dt.fromtimestamp(t) if t + 2082844800 != -9999 else np.nan for t in
                   HK_data['Time_s'][1::] - 2082844800]

    return HK_time_arr, HK_data


def interleave_binary_file(binary_data, channel_format, channel_count, rep_rate_Hz=200000):
    frames = np.array(binary_data)
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

    binary_data_dict['laser_pwr_PT0'] = binary_data_dict['laser_pwr_PT0'] / 100000

    max_cts = rep_rate_Hz / 100

    for channel in list(channel_format):

        if 'counts' in channel:
            binary_data_dict[channel] = np.array([cts if cts < max_cts else np.nan for cts in binary_data_dict[channel]])

            binary_data_dict[channel + '_norm'] = -np.log(1 - (binary_data_dict[channel] / max_cts)) * max_cts

            binary_data_dict[channel + '_norm'] = binary_data_dict[channel + '_norm'] / (binary_data_dict['laser_pwr_PT0'])

    delta_bin = [t_1 - t_0 for t_1, t_0 in zip(binary_data_dict['time_ms'][1::]
                                               , binary_data_dict['time_ms'][0: len(binary_data_dict['time_ms'])])]

    lag_ind = find_min_ind(20 * 1000, delta_bin)

    if delta_bin[lag_ind] > 100:
        lag = delta_bin[lag_ind]
    else:
        lag = np.nan

    return binary_data_dict, lag


def gen_shift_dict(file_name, file_shift, channel_names):
    shift_dict = {name: 0 for name in channel_names}
    # By default the code does not reallign the parameters of any files, unless the below conditional statement is
    # tripped

    if file_name in list(file_shift['name']):
        shift_header = file_shift['col_name'][list(file_shift['name']).index(file_name)]
        shift = file_shift['shift'][list(file_shift['name']).index(file_name)]
        shift_dict[shift_header] = shift
        print('Shifting binary data %s series by %.0f' % (shift_header, shift))

    else:
        print('No file shifting neccessary')

    return shift_dict

