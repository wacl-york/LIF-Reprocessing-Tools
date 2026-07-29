import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def calculate_absolute_humidity(temp, rh):
    # Saturation Vapor Pressure (hPa) using Magnus formula
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    
    # Actual Vapor Pressure (hPa)
    e = es * (rh / 100)
    
    # Absolute Humidity (g/m3)
    # 216.74 is a constant derived from the gas constant for water vapor
    ah = (e * 216.74) / (273.15 + temp)
    
    return ah

metrics_to_compare = ['NO2', 'ozone', 'abs humidity']


# --- read and format NOx data ---
NOx_data = pd.read_csv('C:/Users/pp835/OneDrive - University of York/Documents/Data Analysis/CARES/Mace Head Full Data Analysis/Data/full_processed_datasets/NOx_CARES_MaceHead_prelim_with_baseline_20260113.txt')
NOx_data['Date_time'] = pd.to_datetime(NOx_data['Date_time'])

# --- read and format met data ---
met_data = pd.read_excel('C:/Users/pp835/OneDrive - University of York/Documents/Data Analysis/CARES/Mace Head Full Data Analysis/Data/met_data/VSys_dtLog_May2025.xlsx')
year = 2025
met_data['Date_time'] = (
    pd.to_datetime(f'{year}-01-01') + 
    pd.to_timedelta(met_data['day number'] - 1, unit='D') +  # -1 because Day 1 is Jan 1st
    pd.to_timedelta(met_data['hour'], unit='h') +
    pd.to_timedelta(met_data['min'], unit='m') +
    pd.to_timedelta(met_data['sec'], unit='s')
).dt.ceil('min')

met_data['Abs humidity'] = calculate_absolute_humidity(
    met_data['temperature (deg C)'], 
    met_data['humidity (%)']
)
met_data = met_data.set_index('Date_time')
met_data = met_data[~met_data.index.duplicated(keep='first')]
met_data = met_data.resample('1 min').ffill()
met_data = met_data.reset_index()

# --- read and format o3 data ---
O3_data = pd.read_csv('C:/Users/pp835/OneDrive - University of York/Documents/Data Analysis/CARES/Mace Head Full Data Analysis/Data/met_data/mhd-o3-coast-campaign.dat', sep='\s+')
datetime_string = O3_data['date'].astype(str) + ' ' + O3_data['time'].astype(str)
O3_data['Date_time'] = pd.to_datetime(datetime_string, format='%y%m%d %H:%M')

# --- merge 3 dataframes together ---
comb_data = pd.merge(
    NOx_data
    , met_data
    , on='Date_time'
    , how='left'
    )

comb_data = pd.merge(
    comb_data
    , O3_data
    , on='Date_time'
    , how='left'
    )

# --- set flag for night ---
comb_data['night'] = np.where(
    comb_data['Date_time'].dt.hour <= 2
    ,1
    ,0
    )

# --- isolate nighttime clean NOx ---
night_NOx = comb_data[
    (comb_data['night'] == 1)
    ]

clean_night_NOx = comb_data[
    (comb_data['B'] == 10) &
    (comb_data['night'] == 1)
    ]



# --- plot ---
fig, ax = plt.subplots(4, 1, figsize=(10,20))
ax[0].scatter(clean_night_NOx['wind speed (m/s)'], clean_night_NOx['amb_NO_ppt'])
ax[0].set_ylabel('NO, ppt')
ax[0].set_xlabel('wind speed, ms-1')
ax[0].legend()

ax[1].scatter(clean_night_NOx['amb_NO2_ppt'], clean_night_NOx['amb_NO_ppt'])
ax[1].set_ylabel('NO, ppt')
ax[1].set_xlabel('NO2, ppt')
ax[1].legend()

ax[2].scatter(clean_night_NOx['o3'], clean_night_NOx['amb_NO_ppt'])
ax[2].set_ylabel('NO, ppt')
ax[2].set_xlabel('o3, ppb')
ax[2].legend()

ax[3].scatter(clean_night_NOx['Abs humidity'], clean_night_NOx['amb_NO_ppt'])
ax[3].set_ylabel('NO, ppt')
ax[3].set_xlabel('abs humidity, gm-3')
ax[3].legend()

fig.suptitle('Night time NO correlation'
             )
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(night_NOx['Abs humidity'], night_NOx['amb_NO_ppt'])
ax.set_ylabel('NO, ppt')
ax.set_xlabel('abs humidity, gm-3')
ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, ax1 = plt.subplots(figsize=(10, 8))

line1 = ax1.plot(clean_night_NOx['Date_time'], clean_night_NOx['amb_NO_ppt'], 
                 'o', markersize=2, label='NO, ppt', color='blue')
ax1.set_xlabel('Date_time')
ax1.set_ylabel('NO, ppt', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx() 
line2 = ax2.plot(comb_data['Date_time'], comb_data['Abs humidity'], 
                 label='absolute humidity', color='green')
ax2.set_ylabel('Absolute Humidity (g/m³)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()