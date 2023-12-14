
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime as dt
import allantools as at
from sklearn.metrics import r2_score

from time_stamps import time_average as ta
from main import reprocess_binary_data
from main import process_mix_ratio

reprocess_binary_data(
    bin_file_path='binary data'
    , HK_file_path='HK data'
    , log_start_datetime=['29/11/2023 15:10:00', '30/12/2023 15:10:00', '01/12/2023 15:10:00', '02/12/2023 15:10:00']
)

#breakpoint()

file_list = [f for f in os.listdir('processed data') if os.path.isfile(os.path.join('processed data', f))]
cts_data = pd.DataFrame()

for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv('processed data/' + file)
    ], ignore_index=True)

skip_cal = {
    'start': 100
    , 'end': 60
    , 'tot_sec': 32
}
process_mix_ratio('constant', skip_cal, 4.96)

#fig = make_subplots(specs=[[{"secondary_y": True}]])

#fig.add_trace(go.Scatter(
#    x=[dt.fromtimestamp(t - 2082844800) for t in cts_data['time_s']], y=cts_data['cts_norm']
#    , mode='lines'
#    , name='LIF NO 10 sec / pptv'
#), secondary_y=False)

#fig.update_layout(legend=dict(
#    yanchor="top",
#    y=0.99,
#    xanchor="left",
#    x=0.01
#))
#fig.show()

breakpoint()

file_list = [f for f in os.listdir('mix ratio data') if os.path.isfile(os.path.join('mix ratio data', f))]
mr_data = pd.DataFrame()

for file in file_list:
    mr_data = pd.concat([
        mr_data
        , pd.read_csv('mix ratio data/' + file)
    ], ignore_index=True)

LIF_NO_10Hz = np.array([cts if cts > -8000 else np.nan for cts in mr_data['mix_ratio']])
LIF_60s = ta(mr_data['time_s'], LIF_NO_10Hz, 10, 1/10)
LIF_60s_time_arr = [dt.fromtimestamp(t) for t in LIF_60s[0]]

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(
    x=LIF_60s_time_arr, y=LIF_60s[1]
    , mode='lines'
    , name='LIF NO 10 sec / pptv'
), secondary_y=False)

#fig.update_yaxes(range=[-200, 800], secondary_y=False)
fig.update_yaxes(range=[0.1, 0.4], secondary_y=True)
fig.update_layout(title_text="Comparison of NO Measurements (12/06/2023 - 13/06/2023)"
                  , xaxis_title="Time / BST", yaxis_title="NO mixing ratio / pptv")
#fig.update_layout(yaxis_title="LIF sensitivity / cts mW-1 pptv-1")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

fig.write_html('figures/LIF timeseries.html')
fig.show()