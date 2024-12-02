
import os
import pandas as pd
import plotly.graph_objects as go

from Lifpy.reprocess import setup_LifEnv, reprocess_binary_data


#setup_LifEnv(path=os.path.dirname(__file__))

HK_headers_dict = {
    'task': 'Task'
    , 'cal_mfc': 'Cal_SO2_MFC_Read'
    , 'blc_flag_0': 'BLC_0_flag'
    , 'blc_flag_1': 'BLC_1_flag'
    , 'sig_cell_flow': 'Sig_Cell_Flow'
    , 'cell_pressure': 'Cell_Pressure'
    , 'ref_cell_flow': 'Ref_Cell_SLP'
}

reprocess_binary_data(
    '12/11/2024 13:01:48'
    , HK_headers_dict
    , ignore_first=False
)

file_list = [f for f in os.listdir('data/processed_data') if os.path.isfile(os.path.join('data/processed_data', f))]
cts_data = pd.DataFrame()

for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv('data/processed_data/' + file, header=9)
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

