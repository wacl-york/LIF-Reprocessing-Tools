import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

# --- Configuration and Data Directory ---
data_dir = (
    'C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
    'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\Data'
)

# --- 1. Load and Preprocess NOx Data ---
print("Loading NOx data...")
try:
    nox_file = os.path.join(data_dir, 'NOx_CARES_MaceHead_prelim.txt')
    NOx_data = pd.read_csv(nox_file)
    # Convert 'Date_time' to datetime objects and floor to the minute
    NOx_data['Date_time'] = (
        pd.to_datetime(NOx_data['Date_time']).dt.floor('min')
    )
except FileNotFoundError:
    print(
        f"Error: NOx file not found in {data_dir}. "
        "Please check the path and filename."
    )
    raise

# --- 2. Load and Preprocess Flag Data (Currently unused in the final merge) ---
print("Processing Flag data...")
try:
    flag_file = os.path.join(data_dir, 'mace_head_flag.csv')
    flag_data = pd.read_csv(flag_file)
    flag_data.columns = flag_data.columns.str.strip()
    
    # Rename and convert time column
    time_col = 'Datetime (Start of hour)'
    flag_data['Date_time'] = pd.to_datetime(flag_data[time_col], 
                                            format='mixed')
    flag_data = flag_data.rename(
        columns={'CLEAN = 0 (FLAG=0 Traj=0)': 'clean_flag'}
    )
    
    # Forward fill trajectory data
    traj_col = 'Trajectory (120 hr -clean = 0)'
    flag_data[traj_col] = flag_data[traj_col].ffill()
    
    # Calculate the 'clean_flag' based on FLAG and Trajectory conditions
    flag_data['clean_flag'] = np.where(
        (flag_data['FLAG (WS=0, WD=0, BC=0)'] == 0) &
        (flag_data[traj_col] == 0), 
        0, 
        np.nan
    )
    flag_data = flag_data.set_index('Date_time')
    flag_data_upsampled = flag_data['clean_flag'].resample('min').ffill()
    flag_data_upsampled = flag_data_upsampled.reset_index()

except FileNotFoundError:
    print(f"Error: Flag file not found in {data_dir}.")
except Exception as e:
    print(f"An error occurred during flag data processing: {e}")

# --- 3. Load and Preprocess Baseline Data (Crucial for filtering) ---
print("Loading and processing Baseline data...")
try:
    baseline_file = os.path.join(data_dir, 'MH_G_baseComb2_2025.txt')
    baseline_data = pd.read_csv(
        baseline_file, sep='\s+', header=6, engine='python'
    )
    
    # Coerce time columns to integer, filling NaNs with 0
    for col in ['YY', 'MM', 'DD', 'HH', 'Mn']:
        baseline_data[col] = baseline_data[col].fillna(0).astype(int)
    
    # Create combined datetime string
    datetime_string_series = (
        baseline_data['YY'].astype(str) + '/' +
        baseline_data['MM'].astype(str) + '/' +
        baseline_data['DD'].astype(str) + ' ' +
        baseline_data['HH'].astype(str) + ':' +
        baseline_data['Mn'].astype(str)
    )
    # Convert to datetime objects
    baseline_data['Date_time'] = pd.to_datetime(
        datetime_string_series, errors='coerce'
    )
    
    # Upsample the baseline 'B' flag to minute resolution using forward fill
    baseline_data = baseline_data.set_index('Date_time')
    baseline_data_upsampled = baseline_data['B'].resample('min').ffill()
    baseline_data_upsampled = baseline_data_upsampled.reset_index()

except FileNotFoundError:
    print(f"Error: Baseline file not found in {data_dir}.")
    raise

# --- 4. Merge DataFrames and Apply Baseline Filter (B=10) ---
print("Merging data and applying baseline filter...")
NOx_data = pd.merge(NOx_data, baseline_data_upsampled, 
                    on='Date_time', how='left')

# Filter for baseline conditions where 'B' is exactly 10, otherwise NaN
NOx_data['clean_NO'] = np.where(
    NOx_data['B'] == 10, 
    NOx_data['NO_amb_ppt'], 
    np.nan
)
NOx_data['clean_NOx'] = np.where(
    NOx_data['B'] == 10, 
    NOx_data['NOx_amb_ppt'], 
    np.nan
)

# Calculate clean NO2 concentration (NO2 = NOx - NO)
NOx_data['clean_NO2'] = NOx_data['clean_NOx'] - NOx_data['clean_NO']

# NEW: Calculate the raw NO2 concentration for the 'unclean' data plot
NOx_data['NO2_amb_ppt'] = (
    NOx_data['NOx_amb_ppt'] - NOx_data['NO_amb_ppt']
)

# --- 4.5. Outlier/Spike Removal using IQR Method (3.0 * IQR) ---
print("Removing obvious spikes using 3.0 * IQR filter...")

def iqr_outlier_filter(series, iqr_factor=3.0):
    """Removes outliers using the Interquartile Range (IQR) method."""
    series_clean = series.dropna()
    if len(series_clean) < 2:
        # Not enough data to calculate IQR, return original series
        return series
        
    Q1 = series_clean.quantile(0.25)
    Q3 = series_clean.quantile(0.75)
    IQR = Q3 - Q1
    # Handle case where IQR is zero to prevent issues
    if IQR == 0:
        return series
        
    upper_bound = Q3 + iqr_factor * IQR
    lower_bound = Q1 - iqr_factor * IQR
    # Retain data only within the bounds, set outliers to NaN
    return series.where(
        (series >= lower_bound) & (series <= upper_bound), np.nan
    )

# Apply filtering to the clean columns
NOx_data['clean_NO'] = iqr_outlier_filter(NOx_data['clean_NO'])
NOx_data['clean_NOx'] = iqr_outlier_filter(NOx_data['clean_NOx'])
# The clean_NO2 column is filtered again to remove any resultant outliers 
# that may have been created during the subtraction.
NOx_data['clean_NO2'] = iqr_outlier_filter(NOx_data['clean_NO2'])


# --- 5. Resample, Aggregate, and Calculate Diurnal Medians and IQR ---
print("Calculating 60-minute means, diurnal medians, and IQR...")
# Set index and resample to 60-minute intervals
NOx_data = NOx_data.set_index('Date_time')

# Step 1: Calculate the mean concentration for every 60-minute block.
NOx_data_60min_means = NOx_data[
    ['clean_NO', 'clean_NOx', 'clean_NO2']
].resample('60 min').mean()

# Step 2: Group the 60-minute means by time of day and calculate stats
diurnal_df = NOx_data_60min_means.groupby(
    NOx_data_60min_means.index.time
).agg(
    [
        'median',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]
)

# Rename columns for easier access
diurnal_df.columns = ['_'.join(col).strip() for col in diurnal_df.columns.values]

diurnal_df = diurnal_df.reset_index()
diurnal_df = diurnal_df.rename(columns={'index': 'time'})

# Create a time_delta column for plotting on a continuous axis
diurnal_df['time_delta'] = diurnal_df['time'].apply(
    lambda t: (
        pd.to_timedelta(t.hour, unit='h') + 
        pd.to_timedelta(t.minute, unit='m') + 
        pd.to_timedelta(t.second, unit='s')
    )
)

# Convert time_delta to total hours (float) for numerical plotting
diurnal_df['time_hours'] = (
    diurnal_df['time_delta'].dt.total_seconds() / 3600.0
)


# --- 6. Plotting the Diurnal Cycles with IQR Shading ---
print("Generating separate plots for NO and NO2 with IQR shading (Diurnal Cycle)...")

# Create a figure with two subplots, stacked vertically (2 rows, 1 column)
fig_diurnal, ax_diurnal = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True
) 

# --- Common Variables for Plotting ---
x_data = diurnal_df['time_hours']

# --- Plot 1: Clean NO (Median and IQR Shading) ---
median_no = diurnal_df['clean_NO_median']
q25_no = diurnal_df['clean_NO_q25']
q75_no = diurnal_df['clean_NO_q75']

# Plot the median line
ax_diurnal[0].plot(x_data, median_no, 
                   label='Median Clean NO', 
                   color='#4f46e5', 
                   linewidth=2)

# Add the shading (IQR) using plt.fill_between
ax_diurnal[0].fill_between(x_data, q25_no, q75_no, 
                           color='#4f46e5', 
                           alpha=0.3, 
                           label='IQR (25th to 75th Percentile)')

ax_diurnal[0].set_ylabel('Median NO Concentration (ppt)', fontsize=12)
ax_diurnal[0].set_title(
    'Diurnal Cycle of Clean NO at Mace Head (Median and IQR)', 
    fontsize=14, 
    fontweight='bold'
)
ax_diurnal[0].legend(
    frameon=True, shadow=True, fancybox=True, fontsize=10
)
ax_diurnal[0].set_xlim(0, 24)

# --- Plot 2: Clean NO2 (Median and IQR Shading) ---
median_no2 = diurnal_df['clean_NO2_median']
q25_no2 = diurnal_df['clean_NO2_q25']
q75_no2 = diurnal_df['clean_NO2_q75']

# Plot the median line
ax_diurnal[1].plot(x_data, median_no2, 
                   label='Median Clean $\\text{NO}_2$', 
                   color='#dc2626', 
                   linestyle='-', 
                   linewidth=2)

# Add the shading (IQR) using plt.fill_between
ax_diurnal[1].fill_between(x_data, q25_no2, q75_no2, 
                           color='#dc2626', 
                           alpha=0.3, 
                           label='IQR (25th to 75th Percentile)')

ax_diurnal[1].set_ylabel(
    'Median $\\text{NO}_2$ Concentration (ppt)', fontsize=12
)
ax_diurnal[1].set_title(
    'Diurnal Cycle of Clean NO\u2082 at Mace Head (Median and IQR)', 
    fontsize=14, 
    fontweight='bold'
)
ax_diurnal[1].legend(
    frameon=True, shadow=True, fancybox=True, fontsize=10
)
ax_diurnal[1].set_xlim(0, 24)


# Formatting the X-axis (shared for both plots)
hours_in_day_ticks = np.arange(0, 24, 3)
ax_diurnal[1].set_xticks(hours_in_day_ticks)
ax_diurnal[1].set_xticklabels([f'{h:02d}:00' for h in hours_in_day_ticks])
ax_diurnal[1].set_xlabel('Time of Day (UTC)', fontsize=12)

fig_diurnal.tight_layout() # Adjust layout to prevent overlapping elements


# --- 7. Plotting the Full Time Series Data (Clean vs. Unclean) ---
print("Generating full time series plots (60-min means) showing clean "
      "and other data...")

# 'Other' data is where the B flag is NOT 10
NOx_data['other_NO'] = np.where(
    NOx_data['B'] != 10, 
    NOx_data['NO_amb_ppt'], 
    np.nan
)
NOx_data['other_NO2'] = np.where(
    NOx_data['B'] != 10, 
    NOx_data['NO2_amb_ppt'], 
    np.nan
)

# Resample all plotting columns to 60-minute means
plot_data_60min = NOx_data[
    ['clean_NO', 'clean_NO2', 'other_NO', 'other_NO2']
].resample('60 min').mean()


# Create a second figure for the time series
fig_timeseries, ax_timeseries = plt.subplots(
    2, 1, figsize=(15, 8), sharex=True
)

# --- Plot 1: NO Concentration ---
# Plot Clean NO (Emerald Green)
ax_timeseries[0].plot(plot_data_60min.index, 
                      plot_data_60min['clean_NO'], 
                      color='#10b981', 
                      linewidth=1.5, 
                      label='Clean NO (B=10)') 
# Plot Other NO (Red)
ax_timeseries[0].plot(plot_data_60min.index, 
                      plot_data_60min['other_NO'], 
                      color='#ef4444', 
                      linewidth=1.5, 
                      label='Other Data (B \u2260 10)') 

ax_timeseries[0].set_title(
    'Full Time Series of NO Concentration at Mace Head (60-min Mean)', 
    fontsize=14, 
    fontweight='bold'
)
ax_timeseries[0].set_ylabel('NO Concentration (ppt)', fontsize=12)
ax_timeseries[0].legend(loc='upper right')
ax_timeseries[0].grid(True, linestyle=':', alpha=0.6)


# --- Plot 2: NO2 Concentration ---
# Plot Clean NO2 (Emerald Green)
ax_timeseries[1].plot(plot_data_60min.index, 
                      plot_data_60min['clean_NO2'], 
                      color='#10b981', 
                      linewidth=1.5, 
                      label='Clean $\\text{NO}_2$ (B=10)')
# Plot Other NO2 (Red)
ax_timeseries[1].plot(plot_data_60min.index, 
                      plot_data_60min['other_NO2'], 
                      color='#ef4444', 
                      linewidth=1.5, 
                      label='Other Data (B \u2260 10)')

ax_timeseries[1].set_title(
    'Full Time Series of $\\text{NO}_2$ Concentration at Mace Head (60-min Mean)', 
    fontsize=14, 
    fontweight='bold'
)
ax_timeseries[1].set_ylabel(
    '$\\text{NO}_2$ Concentration (ppt)', fontsize=12
)
ax_timeseries[1].legend(loc='upper right')
ax_timeseries[1].grid(True, linestyle=':', alpha=0.6)


# Formatting the X-axis (shared for both plots)
# Use a formatter for month and year visibility
date_form = DateFormatter("%Y-%m")
ax_timeseries[1].xaxis.set_major_formatter(date_form)
ax_timeseries[1].xaxis.set_major_locator(MonthLocator(interval=2))
ax_timeseries[1].set_xlabel('Date (Year-Month)', fontsize=12)

fig_timeseries.tight_layout() # Adjust layout for the second figure

plt.show() # Display both figures

print("All plots generated successfully: Diurnal cycle (Median and IQR) and "
      "full campaign Time Series (60-min mean) showing clean vs. other data.")