
import os
import pandas as pd
import plotly.graph_objects as go

from LifPy.reprocess import setup_LifEnv, reprocess_binary_data


setup_LifEnv(path=os.path.dirname(__file__))

HK_headers_dict = {
    'task': 'Task'
    , 'cal_mfc': 'Cal_SO2_MFC_Read'
    , 'blc_flag_0': 'BLC_0_flag'
    , 'blc_flag_1': 'BLC_1_flag'
    , 'sig_cell_flow': 'Sig_Cell_Flow'
    , 'cell_pressure': 'Cell_Pressure'
    , 'ref_cell_flow': 'Ref_Cell_SLPM'
}

channel_format = {
    'sig_counts_A': 0
    , 'ref_counts': 1
    , 'seed_LD_current': 2
    , 'laser_pwr_PT0': [3, 4]
    , 'time_ms': [7, 8]
    , 'seed_LD_mode': 9
    #, 'sig_counts_B': 10
}  # if your data is a form of signal counts it must follow the format 'sig_counts_X' where X can be any letter

reprocess_binary_data(
    '29/05/2025 08:58:00'
    , HK_headers_dict
    , channel_format
    , ignore_first=False
    , channel_count=10
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
    x=cts_data['mac_time_s'], y=cts_data['sig_A_diff_cts'] / cts_data['ref_cts_diff']
    , mode='lines'
))

fig.add_trace(go.Scatter(
    x=cts_data['mac_time_s'], y=cts_data['cal_mfc']
    , mode='lines'
))

fig.write_html('sample.html')
fig.show()

