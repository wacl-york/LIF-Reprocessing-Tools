import numpy as np
import math
from datetime import datetime as dt


def time_average(time_array, data_array, current_rate_Hz, desired_rate_Hz):

    if current_rate_Hz < desired_rate_Hz:
        return time_array, data_array, 'Cannot time average, reduced desired_rate_Hz'

    if current_rate_Hz == desired_rate_Hz:
        return time_array, data_array, 'No change to array dimensions'

    if current_rate_Hz > desired_rate_Hz:

        new_array_len = math.floor(len(time_array) / (current_rate_Hz / desired_rate_Hz))

        new_time_arr = []
        new_data_arr = []
        for i in range(new_array_len - 1):
            index = int(i * (current_rate_Hz / desired_rate_Hz))

            new_time_arr.append(np.mean(time_array[index: index + int(current_rate_Hz / desired_rate_Hz)]))
            new_data_arr.append(np.mean(data_array[index: index + int(current_rate_Hz / desired_rate_Hz)]))

        return new_time_arr, new_data_arr, new_array_len

def find_min_ind(target, array):
    diff_arr = []
    for i in range(len(array)):
        diff_arr.append(abs(array[i] - target))

    return diff_arr.index(np.min(diff_arr))


def index_timestamp_1904(time_string, time_array):
    time_float = dt.timestamp(dt.strptime(time_string, '%d/%m/%Y %H:%M:%S'))

    epoch_dif = (dt.strptime('01/01/1970', '%d/%m/%Y') - dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()

    time_index = find_min_ind(time_float + epoch_dif, time_array)

    return time_index
