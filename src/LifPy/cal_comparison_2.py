import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from typing import Dict, Any

# --- Configuration and Data Directory ---
data_dir = (
    'C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
    'Data Analysis\\CARES\\Mace Head Full Data Analysis\\Data'
)

# --- 1. Helper Function: IQR Outlier Filter ---
def iqr_outlier_filter(series: pd.Series, iqr_factor: float = 3.0) -> pd.Series:
    """Removes outliers using the Interquartile Range (IQR) method."""
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return series
        
    Q1 = series_clean.quantile(0.25)
    Q3 = series_clean.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return series
        
    upper_bound = Q3 + iqr_factor * IQR
    lower_bound = Q1 - iqr_factor * IQR
    
    # Retain data only within the bounds, set outliers to NaN
    return series.where(
        (series >= lower_bound) & (series <= upper_bound), np.nan
    )

# --- 2. Main Processing Function ---
def process_and_calculate_diurnal(
    nox_data_path: str, 
    baseline_data_path: str, 
    flag_data_path: str
) -> Dict[str, pd.DataFrame]:
    """
    Loads and preprocesses NOx, Baseline data, calculates 60-minute means, 
    and derives the diurnal cycle (median/IQR).
    
    Returns a dict containing the diurnal DataFrame and 60-min time series.
    """
    print(f"\n--- Starting Processing for {os.path.basename(nox_data_path)} ---")
    
    # 2.1 Load and Preprocess NOx Data
    try:
        NOx_data = pd.read_csv(nox_data_path)
        NOx_data['Date_time'] = (
            pd.to_datetime(NOx_data['Date_time']).dt.floor('min')
        )
    except FileNotFoundError:
        print(f"Error: NOx file not found at {nox_data_path}. Skipping.")
        return None
    
    # 2.2 Load and Preprocess Baseline Data
    try:
        baseline_data = pd.read_csv(
            baseline_data_path, sep='\s+', header=6, engine='python'
        )
        
        for col in ['YY', 'MM', 'DD', 'HH', 'Mn']:
            baseline_data[col] = baseline_data[col].fillna(0).astype(int)
            
        datetime_string_series = (
            baseline_data['YY'].astype(str) + '/' +
            baseline_data['MM'].astype(str) + '/' +
            baseline_data['DD'].astype(str) + ' ' +
            baseline_data['HH'].astype(str) + ':' +
            baseline_data['Mn'].astype(str)
        )
        baseline_data['Date_time'] = pd.to_datetime(
            datetime_string_series, errors='coerce'
        )
        
        baseline_data = baseline_data.set_index('Date_time')
        baseline_data_upsampled = baseline_data['B'].resample('min').ffill()
        baseline_data_upsampled = baseline_data_upsampled.reset_index()

    except FileNotFoundError:
        print(f"Error: Baseline file not found at {baseline_data_path}. Skipping.")
        return None
        
    # 2.3 Merge and Apply Baseline Filter (B=10)
    NOx_data = pd.merge(NOx_data, baseline_data_upsampled,  
                        on='Date_time', how='left')

    # Filter for baseline conditions where 'B' is exactly 10, otherwise NaN
    NOx_data['clean_NO'] = np.where(
        NOx_data['B'] == 10,  
        NOx_data['amb_NO_ppt'],  
        np.nan
    )
    NOx_data['clean_NO2'] = np.where(
        NOx_data['B'] == 10,  
        NOx_data['amb_NO2_ppt'],  
        np.nan
    )
    
    # 2.4 Outlier/Spike Removal
    NOx_data['clean_NO'] = iqr_outlier_filter(NOx_data['clean_NO'])
    NOx_data['clean_NO2'] = iqr_outlier_filter(NOx_data['clean_NO2'])
    
    # 2.5 Calculate 'other' data (B != 10) for time series plotting
    NOx_data['other_NO'] = np.where(
        NOx_data['B'] != 10, NOx_data['amb_NO_ppt'], np.nan
    )
    NOx_data['other_NO2'] = np.where(
        NOx_data['B'] != 10, NOx_data['amb_NO2_ppt'], np.nan
    )

    # 2.6 Resample to 60-minute means
    NOx_data = NOx_data.set_index('Date_time')
    NOx_data_60min = NOx_data[
        ['clean_NO', 'clean_NO2', 'other_NO', 'other_NO2']
    ].resample('60 min').mean()
    
    # 2.7 Calculate Diurnal Medians and IQR
    diurnal_df = NOx_data_60min.groupby(
        NOx_data_60min.index.time
    ).agg(
        ['median', ('q25', lambda x: x.quantile(0.25)), 
         ('q75', lambda x: x.quantile(0.75))]
    )

    diurnal_df.columns = ['_'.join(col).strip() for col in diurnal_df.columns.values]
    diurnal_df = diurnal_df.reset_index().rename(columns={'index': 'time'})

    diurnal_df['time_hours'] = diurnal_df['time'].apply(
        lambda t: (pd.to_timedelta(t.hour, unit='h') + 
                   pd.to_timedelta(t.minute, unit='m')
                  ).total_seconds() / 3600.0
    )
    
    return {
        'diurnal': diurnal_df, 
        'timeseries_60min': NOx_data_60min
    }

# =====================================================================
# --- MAIN EXECUTION BLOCK ---
# =====================================================================

# --- Define File Paths for Two Datasets (Keys match function parameters) ---
dataset_paths = {
    'Dataset 1': {
        'nox_data_path': os.path.join(data_dir, 'file_save_test_data.txt'),
        'baseline_data_path': os.path.join(data_dir, 'MH_G_baseComb2_2025.txt'),
        'flag_data_path': os.path.join(data_dir, 'mace_head_flag.csv')
    },
    'Dataset 2': {
        'nox_data_path': os.path.join(data_dir, 'NOx_CARES_MaceHead_prelim_20251209.txt'), 
        'baseline_data_path': os.path.join(data_dir, 'MH_G_baseComb2_2025.txt'),
        'flag_data_path': os.path.join(data_dir, 'mace_head_flag.csv')
    }
}

# --- Process Data ---
data1_results = process_and_calculate_diurnal(**dataset_paths['Dataset 1'])
data2_results = process_and_calculate_diurnal(**dataset_paths['Dataset 2'])

if data1_results is None or data2_results is None:
    print("One or more datasets could not be processed. Stopping.")
    exit()

ts_df1 = data1_results['timeseries_60min']
ts_df2 = data2_results['timeseries_60min']

# =====================================================================
# --- 5. Calculate Average Percentage Difference ---
# =====================================================================
print("\nCalculating average percentage difference...")

# Merge the clean concentration columns on the time index
comparison_df = pd.merge(
    ts_df1[['clean_NO', 'clean_NO2']], 
    ts_df2[['clean_NO', 'clean_NO2']], 
    left_index=True, 
    right_index=True, 
    suffixes=('_D1', '_D2')
).dropna() # Drop rows where either dataset has NaN (no clean measurement)

def calculate_avg_abs_pct_diff(series_d1, series_d2):
    """Calculates the average absolute percentage difference."""
    # Absolute difference
    diff = np.abs(series_d1 - series_d2)
    # Average concentration (denominator)
    avg_conc = (series_d1 + series_d2) / 2
    
    # Calculate percentage difference, avoiding division by zero
    # np.divide is used to handle divide-by-zero, returning NaN where avg_conc is 0
    pct_diff = np.divide(diff, avg_conc, out=np.full_like(diff, np.nan), where=avg_conc != 0) * 100
    
    # Return the mean of the absolute percentage differences
    return pct_diff.mean()

# Calculate the average difference for NO and NO2
avg_pct_diff_NO = calculate_avg_abs_pct_diff(
    comparison_df['clean_NO_D1'], comparison_df['clean_NO_D2']
)
avg_pct_diff_NO2 = calculate_avg_abs_pct_diff(
    comparison_df['clean_NO2_D1'], comparison_df['clean_NO2_D2']
)

print(f"Average Absolute Percentage Difference (Clean NO): {avg_pct_diff_NO:.2f}%")
print(f"Average Absolute Percentage Difference (Clean NO2): {avg_pct_diff_NO2:.2f}%")

# Continue with plotting...
diurnal_df1 = data1_results['diurnal']
diurnal_df2 = data2_results['diurnal']

# =====================================================================
# --- 3. Plotting the Diurnal Cycles (Comparison) ---
# ... (Plotting code for diurnal cycles remains the same)
# =====================================================================
print("\nGenerating comparative diurnal plots...")

# Define comparison colors
COLOR1 = '#4f46e5'  # Indigo for Dataset 1
COLOR2 = '#0d9488'  # Teal for Dataset 2

fig_diurnal, ax_diurnal = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True
) 
x_data = diurnal_df1['time_hours'] 

# --- Plot 1: Clean NO Comparison ---
ax_diurnal[0].plot(x_data, diurnal_df1['clean_NO_median'], 
                   label='Median NO (Dataset 1)', 
                   color=COLOR1, linewidth=2)
ax_diurnal[0].fill_between(x_data, diurnal_df1['clean_NO_q25'], 
                           diurnal_df1['clean_NO_q75'], 
                           color=COLOR1, alpha=0.3, label='_nolegend_')
ax_diurnal[0].plot(x_data, diurnal_df2['clean_NO_median'], 
                   label='Median NO (Dataset 2)', 
                   color=COLOR2, linewidth=2, linestyle='--')
ax_diurnal[0].fill_between(x_data, diurnal_df2['clean_NO_q25'], 
                           diurnal_df2['clean_NO_q75'], 
                           color=COLOR2, alpha=0.15, label='_nolegend_')

ax_diurnal[0].set_ylabel('Median NO Concentration (ppt)', fontsize=12)
ax_diurnal[0].set_title(
    'Diurnal Cycle of Clean NO Comparison (Median and IQR)', 
    fontsize=14, 
    fontweight='bold'
)
ax_diurnal[0].legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
ax_diurnal[0].set_xlim(0, 24)

# --- Plot 2: Clean NO2 Comparison ---
ax_diurnal[1].plot(x_data, diurnal_df1['clean_NO2_median'], 
                   label='Median $\\text{NO}_2$ (Dataset 1)', 
                   color=COLOR1, linewidth=2)
ax_diurnal[1].fill_between(x_data, diurnal_df1['clean_NO2_q25'], 
                           diurnal_df1['clean_NO2_q75'], 
                           color=COLOR1, alpha=0.3, label='_nolegend_')
ax_diurnal[1].plot(x_data, diurnal_df2['clean_NO2_median'], 
                   label='Median $\\text{NO}_2$ (Dataset 2)', 
                   color=COLOR2, linewidth=2, linestyle='--')
ax_diurnal[1].fill_between(x_data, diurnal_df2['clean_NO2_q25'], 
                           diurnal_df2['clean_NO2_q75'], 
                           color=COLOR2, alpha=0.15, label='_nolegend_')

ax_diurnal[1].set_ylabel('Median $\\text{NO}_2$ Concentration (ppt)', fontsize=12)
ax_diurnal[1].set_title(
    'Diurnal Cycle of Clean $\\text{NO}_2$ Comparison (Median and IQR)', 
    fontsize=14, 
    fontweight='bold'
)
ax_diurnal[1].legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
ax_diurnal[1].set_xlim(0, 24)

# Formatting the X-axis (shared for both plots)
hours_in_day_ticks = np.arange(0, 24, 3)
ax_diurnal[1].set_xticks(hours_in_day_ticks)
ax_diurnal[1].set_xticklabels([f'{h:02d}:00' for h in hours_in_day_ticks])
ax_diurnal[1].set_xlabel('Time of Day (UTC)', fontsize=12)
fig_diurnal.tight_layout()

# =====================================================================
# --- 4. Plotting the Full Time Series Data (Clean vs. Other Comparison) ---
# ... (Plotting code for time series remains the same)
# =====================================================================
print("Generating comparative full time series plots (Clean vs. Other)...")

fig_timeseries, ax_timeseries = plt.subplots(
    2, 1, figsize=(15, 8), sharex=True
)

# Define comparison colors
COLOR_CLEAN_1 = '#4f46e5'  # Indigo for Clean Data 1
COLOR_OTHER_1 = '#ef4444'  # Red for Other Data 1
COLOR_CLEAN_2 = '#0d9488'  # Teal for Clean Data 2
COLOR_OTHER_2 = '#cc5800'  # Orange for Other Data 2

# --- Plot 1: NO Concentration Comparison (Clean vs. Other) ---
ax_timeseries[0].plot(ts_df1.index, ts_df1['clean_NO'], 
                      color=COLOR_CLEAN_1, linewidth=1.5, 
                      label='Clean NO (Data 1)') 
ax_timeseries[0].plot(ts_df1.index, ts_df1['other_NO'], 
                      color=COLOR_OTHER_1, linewidth=1, linestyle=':',
                      label='Other NO (Data 1)') 
ax_timeseries[0].plot(ts_df2.index, ts_df2['clean_NO'], 
                      color=COLOR_CLEAN_2, linewidth=1.5, linestyle='--',
                      label='Clean NO (Data 2)') 
ax_timeseries[0].plot(ts_df2.index, ts_df2['other_NO'], 
                      color=COLOR_OTHER_2, linewidth=1, linestyle='-.',
                      label='Other NO (Data 2)')

ax_timeseries[0].set_title(
    'Full Time Series of NO Concentration (Clean vs. Other Times)', 
    fontsize=14, fontweight='bold'
)
ax_timeseries[0].set_ylabel('NO Concentration (ppt)', fontsize=12)
ax_timeseries[0].legend(loc='upper right', ncol=2)
ax_timeseries[0].grid(True, linestyle=':', alpha=0.6)


# --- Plot 2: NO2 Concentration Comparison (Clean vs. Other) ---
ax_timeseries[1].plot(ts_df1.index, ts_df1['clean_NO2'], 
                      color=COLOR_CLEAN_1, linewidth=1.5, 
                      label='Clean $\\text{NO}_2$ (Data 1)')
ax_timeseries[1].plot(ts_df1.index, ts_df1['other_NO2'], 
                      color=COLOR_OTHER_1, linewidth=1, linestyle=':',
                      label='Other $\\text{NO}_2$ (Data 1)')
ax_timeseries[1].plot(ts_df2.index, ts_df2['clean_NO2'], 
                      color=COLOR_CLEAN_2, linewidth=1.5, linestyle='--',
                      label='Clean $\\text{NO}_2$ (Data 2)')
ax_timeseries[1].plot(ts_df2.index, ts_df2['other_NO2'], 
                      color=COLOR_OTHER_2, linewidth=1, linestyle='-.',
                      label='Other $\\text{NO}_2$ (Data 2)')

ax_timeseries[1].set_title(
    'Full Time Series of $\\text{NO}_2$ Concentration (Clean vs. Other Times)', 
    fontsize=14, fontweight='bold'
)
ax_timeseries[1].set_ylabel('$\\text{NO}_2$ Concentration (ppt)', fontsize=12)
ax_timeseries[1].legend(loc='upper right', ncol=2)
ax_timeseries[1].grid(True, linestyle=':', alpha=0.6)


# Formatting the X-axis (shared for both plots)
date_form = DateFormatter("%Y-%m")
ax_timeseries[1].xaxis.set_major_formatter(date_form)
ax_timeseries[1].xaxis.set_major_locator(MonthLocator(interval=2))
ax_timeseries[1].set_xlabel('Date (Year-Month)', fontsize=12)

fig_timeseries.tight_layout()
plt.show()

print("All comparative plots generated successfully.")