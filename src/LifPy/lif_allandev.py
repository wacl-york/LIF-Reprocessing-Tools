import allantools
import pylab as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go


#setting the path to make locating files for the file list easier
path = 'C:\\Users\\pp835\\OneDrive - University of York\\Documents\\Data Analysis\\CARES\\Post campaign testing\\20250815\\'


#locating the files and creating an empty dataframe to pull data into
file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
cts_data = pd.DataFrame()

#pulling all files from the destination folder into the dataframe that has been set up
for file in file_list:
    cts_data = pd.concat([
        cts_data
        , pd.read_csv(path + file, header=7)
    ], ignore_index=True)
    
cts_data.replace(-9999, np.nan, inplace=True)

cts_data['mac_time_s'] = (cts_data['mac_time_s']-2082844800) # number of seconds 01/01/1904 - 01/01/1970
cts_data['mac_time_s'] = pd.to_datetime(cts_data['mac_time_s'], unit = 's')

cts_data = cts_data.set_index('mac_time_s')
cts_data = cts_data.resample('1s').mean()
cts_data = cts_data.reset_index()

cts_data_filtered = cts_data[cts_data.index >= 475].copy()
cts_data_filtered = cts_data.reset_index(drop=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
     x = cts_data_filtered['mac_time_s'],  y = cts_data_filtered['sig_A_diff_cts']/cts_data['ref_cts_diff']
     , mode = 'markers'
))
fig.write_html('allandev.html')
fig.show() 
    
    
t = np.logspace(-1, 3, 100)  # tau values from 0.1 to 10000
y = cts_data_filtered['sig_A_diff_cts']/cts_data_filtered['ref_cts_diff']  
r = 10  # sample rate in Hz of the input data
(t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # Compute the overlapping ADEV
fig = plt.loglog(t2, ad) # Plot the results
# plt.show()

cts_data_filtered = cts_data_filtered.set_index('mac_time_s')
cts_data_filtered = cts_data_filtered.resample('10s').mean()
print('sd B = ' + str(np.std(cts_data_filtered['sig_B_diff_cts']/cts_data_filtered['ref_cts_diff'])))
print('LOD B = 3 x sd = ' + str(3*np.std(cts_data_filtered['sig_B_diff_cts']/cts_data_filtered['ref_cts_diff'])))