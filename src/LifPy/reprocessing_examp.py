
import os
import pandas as pd
import plotly.graph_objects as go

from LifPy.reprocess import reprocess_binary_data

reprocess_binary_data(
    log_start_datetime='12/11/2024 13:01:48'
    , ignore_first=False
    , gen_diag_plots=False
)

file_list = [f for f in os.listdir('data/processed_data') if os.path.isfile(os.path.join('data/processed_data', f))]
cts_data = pd.DataFrame()

for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv('data/processed_data/' + file)
    ], ignore_index=True)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=cts_data['mac_time_s'], y=cts_data['sig_cts_diff'] / cts_data['ref_cts_diff']
    , mode='lines'
))

fig.add_trace(go.Scatter(
    x=cts_data['mac_time_s'], y=cts_data['blc_flag_0']
    , mode='lines'
))

fig.add_trace(go.Scatter(
    x=cts_data['mac_time_s'], y=cts_data['blc_flag_1']
    , mode='lines'
))

fig.show()

