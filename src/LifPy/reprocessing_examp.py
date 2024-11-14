
import os
import pandas as pd
import plotly.graph_objects as go

from LifPy import reprocess_binary_data


reprocess_binary_data(
    log_start_datetime='28/10/2024 14:49:33'
    , ignore_first=False
    , config_path=(r'{}/' + 'bin/config.txt').format(os.path.dirname(__file__))
    , gen_diag_plots=False
)

file_list = [f for f in os.listdir('processed data') if os.path.isfile(os.path.join('processed data', f))]
cts_data = pd.DataFrame()

for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv('processed data/' + file)
    ], ignore_index=True)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=cts_data['time_s'], y=cts_data['sig_cts_diff'] / cts_data['ref_cts_diff']
    , mode='lines'
))

fig.show()

