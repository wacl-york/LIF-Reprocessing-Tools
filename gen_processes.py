
import os
import numpy as np


def generate_folder(path):
    # A function to create new folders if they do not already exist. The folders are created in the directory where the
    # function is located and the path should include the name you want to designate to the folder

    dir_path = os.path.dirname(__file__)
    path = (r'{}/' + path).format(dir_path)

    try:
        # The try statement allows for an exception if the folder already exists
        os.makedirs(path)
    except OSError:
        pass

def find_min_ind(target, array):
    # A function which takes an array (MUST BE NUMPY.ARRAY FORMAT) and locates the value in that array which is
    # numerically closest to a user inputed target value. The return is the index of that value in the array

    diff_arr = []
    for i in range(len(array)):
        diff_arr.append(abs(array[i] - target))

    return diff_arr.index(np.min(diff_arr))


def search_tsk(data, flag):

    ind_dict = {}

    for i in range(1, len(data['task'][1::])):
        if data['task'][i] == flag and data['task'][i - 1] != flag:

            for j in range(i, len(data['task'][1::])):
                if data['task'][j] == flag and data['task'][j + 1] != flag:
                    break
            ind_dict[len(list(ind_dict))] = [i, j]

    return ind_dict