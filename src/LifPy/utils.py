
import os
import numpy as np
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

