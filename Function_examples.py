
import os
import pandas as pd
import matplotlib.pyplot as plt

from time_stamps import index_timestamp_1904
from main import reprocess_binary_data, process_mix_ratio

reprocess_binary_data(
    bin_file_path='Binary data'
    , HK_file_path='HK data'
    , log_start_datetime=['29/11/2023 15:10:00', '30/12/2023 15:10:00']
)

file_list = [f for f in os.listdir('processed data') if os.path.isfile(os.path.join('processed data', f))]
cts_data = pd.DataFrame()

for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv('processed data/' + file)
    ], ignore_index=True)

plt.plot(cts_data['cts_norm'])
plt.show()

