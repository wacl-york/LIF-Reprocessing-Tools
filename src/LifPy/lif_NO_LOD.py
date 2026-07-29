import lif_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.odr import ODR, Model, RealData
from scipy.stats import linregress

# --------------------- Variables to manually input --------------------------

molecule = 'NO'                                                                 # NO (even for multi-channel NOx) or SO2
channels = {'sig_A':'NO', 'sig_B':'NOx', 'sig_C':'NOy'}                         # dict of all sig channels
start_times = {'sig_A':'30/05/2025 00:00', 'sig_B':'05/06/2025 00:00'}
cal_task = 5                                                                    # Task number associated with regular cals
BLC_cal_task = 1                                                                # Task number associated with BLC cals
zero_task = 4                                                                   # Task number associated with zeroes
R2_limit = 0.75                                                                 # Lower limit for R2 values of cal regression
BLC = False                                                                     # Boolean indicator of whether BLC present
plot = True                                                                     # diagnostics plots at various analysis stages
averaging = '1s'                                                              # averaging for the final output file
filename = 'NOx_CARES_MaceHead_prelim_20260113'                                 # filename for the resampled MR output
cal_cylinder_conc = 5000000                                                     # in ppt
pre_taskswitch = 300    # (at 10Hz = 30 secs)                                   # data points before task switch to ignore
post_taskswitch = 600   # (at 10Hz = 60 secs)                                   # data points after task switch to ignore
pre_peakfind = 20       # (at 10Hz = 2 secs)                                    # data points before ref_cts_diff drop to ignore
post_peakfind = 200     # (at 10Hz = 20 secs)                                   # data points after ref_cts_diff drop to ignore
ref_cts_diff_limit = 15000                                                      # lower limit ref_diff_cts_norm

# data_dir = (r'C:/Users/pp835/OneDrive - University of York/Documents/Data Analysis/'
#             'CARES/Mace Head Full Data Analysis/Data/SNR Test'
#             )

data_dir = (r'C:/Users/pp835/OneDrive - University of York/Documents/'
            'Data Analysis/CARES/Post campaign testing/troubleshooting_data'
            )
# ----------------------------------------------------------------------------


day_folders = lif.find_day_folders(data_dir)
cts_data = lif.read_processed_files(
    data_dir, day_folders
    )

def calc_LOD(data, data_dir, channels, molecule, cal_cylinder_conc, plot, cal_task, zero_task):
    """
    NO-LIF LOD calculation:
    """
    print('\nStarting LOD Calculation')
    
    cts_data = data.copy()
    cts_data_mask = cts_data['Date_time'] > '2026-05-06 10:10:00'
    cts_data = cts_data[cts_data_mask]
    
    # ---  Flow and cal conc calculation ---
    cal_mfc_slpm = cts_data[f'Cal_{molecule}_MFC_Read'] / 1000.0
    flows = [col for col in cts_data.columns if 'Flow' in col]
    cts_data['total_flow'] = cts_data[flows].sum(axis=1) 
    cts_data[f'{molecule}_mr'] = (cal_mfc_slpm / cts_data['total_flow']) * cal_cylinder_conc # gives NO in ppt

    # ---  ref normalisation ---
    for channel in channels:
        chan_col = f'{channel}_diff_cts_ref_norm'
        cts_data[chan_col] = (
            cts_data[f'{channel}_diff_cts_norm'] / cts_data['ref_diff_cts_norm']
        )

    # ---  Zero task isolation ---
    zero_mask = cts_data['Task'] == zero_task
    zero_data = cts_data[zero_mask]
    
    if zero_data.empty:
        print(f"WARNING: No data found for zero_task: '{zero_task}'")

    # ---  Cal Task Filtering ---
    cts_data['start_of_cal'] = np.where(
        (cts_data['Task'] == cal_task) & (cts_data['Task'].shift(1) != cal_task), 1, 0
    )
    cts_data['cal_number'] = cts_data['start_of_cal'].cumsum()
    cts_data = cts_data[(cts_data['Task'] == cal_task) & (cts_data['Cal_SB_MFC_Read'] < 0.1)].copy()


    for channel in channels:
        chan_col = f'{channel}_diff_cts_ref_norm'
        cts_data[chan_col] = (
        cts_data[f'{channel}_diff_cts_norm'] / (cts_data['ref_diff_cts_norm'])
        )
        print(f'\n--- Processing Channel: {channel} ---')

        for cal_num, cal_df in cts_data.groupby('cal_number'):
            if len(cal_df) < 15: continue
            
            cal_df['cal_point'] = (cal_df[f'Cal_{molecule}_MFC_set'] != cal_df[f'Cal_{molecule}_MFC_set'].shift(1)).cumsum()
            
            # --- Trimming cal points to filter transient MFC periods ---
            SNR_results_list = []
            cal_df['stable_cal_point'] = False

            for cp_num, cp_df in cal_df.groupby('cal_point'):
                n = len(cp_df)
                trim = int(np.ceil(n * 0.05)) 
                
                if n > (2 * trim):
                    stable_slice = cp_df.iloc[trim : n-trim].copy()
                    cal_df.loc[stable_slice.index, 'stable_cal_point'] = True
                    
                    sig_mean = stable_slice[chan_col].mean()
                    noise_std = stable_slice[chan_col].std()
                    
                    SNR_results_list.append({'signal': sig_mean, 'noise': noise_std})
                    
            stable_cal_point_mask = cal_df['stable_cal_point'] == True
            cal_df_filtered = cal_df[stable_cal_point_mask].copy()

            regression_cols = [f'{molecule}_mr',
                               chan_col]
            cal_df_cleaned = cal_df_filtered.replace(
                [np.inf, -np.inf], np.nan).dropna(
                    subset=regression_cols)

            # --- Prepare data for regression ---
            X = cal_df_cleaned[f'{molecule}_mr']
            Y = cal_df_cleaned[chan_col]
            
            if len(cal_df_filtered) > 1:
                
                # 1. Group the filtered data by the individual stable calibration steps
                cal_points_filtered = cal_df_filtered.groupby('cal_point')

                # 2. Calculate the standard deviation (noise) for X and Y in each stable point
                point_stds = cal_points_filtered.agg({
                    f'{molecule}_mr': 'std', 
                    chan_col: 'std' 
                }).rename(columns={
                    f'{molecule}_mr': 'sigma_X_point',
                    chan_col: 'sigma_Y_point'
                }).dropna() # Drop points that had too few data points to calculate std

                # 3. Determine the overall representative noise (using the Median)
                sx_auto = point_stds['sigma_X_point'].median()
                sy_auto = point_stds['sigma_Y_point'].median()
                
                # 4. Fallback and Final Safety Check
                # Use the median if available, otherwise fall back to overall std dev of the filtered data
                # Ensure the value is not zero to prevent ODR solver errors
                sx_final = max(sx_auto if not np.isnan(sx_auto) else cal_df_filtered[f'{molecule}_mr'].std(), 1e-12)
                sy_final = max(sy_auto if not np.isnan(sy_auto) else cal_df_filtered[chan_col].std(), 1e-12)

            else:
                # Default values if not enough data to calculate stats
                sx_final = 1e-12
                sy_final = 1e-12
                    
                    
            def linear_model(p, x):
                """
                Linear function for ODR: y = m*x + c
                p is the array of parameters [m (slope), c (intercept)]
                x is the independent variable (MR supplied by MFC)
                """
                m, c = p
                return m*x + c
            
            if len(X) < 2 or X.nunique() < 2:
                print(' filtered data has no points')
                slope, intercept, r_value, p_value, \
                    std_err_of_slope = [np.nan] * 5
            
            else:
                # define ODR model based on linear function
                linear_model_odr = Model(linear_model)   
                
                # Define the data, including errors (often estimated as 1.0 if unknown)
                data_odr = RealData(
                    X, Y, 
                    sx=sx_final, # Estimate of standard deviation/error in X (MR)
                    sy=sy_final  # Estimate of standard deviation/error in Y (Signal Counts)
                )
                
                # Instantiate the ODR solver
                # beta0 = initial guess for [slope, intercept]. Using [1, 0] is a good start.
                odr = ODR(data_odr, linear_model_odr, beta0=[1.0, 0.0])     
                
                # Run the regression
                output = odr.run()
                
                # Extract results
                slope = output.beta[0]
                intercept = output.beta[1]
                slope_err = output.sd_beta[0]
                intercept_err = output.sd_beta[1]
                cov_matrix = output.cov_beta
                
                # Create a sorted range for smooth plotting
                x_fit = np.linspace(X.min(), X.max(), 100)
                y_fit = slope * x_fit + intercept
                
                # Calculate the standard error of the fit at each point x
                # Formula: sigma_y = sqrt(x^2 * var(m) + var(c) + 2 * x * cov(m,c))
                sig_y = np.sqrt(
                    (x_fit**2 * cov_matrix[0, 0]) + 
                    cov_matrix[1, 1] + 
                    (2 * x_fit * cov_matrix[0, 1])
                )
                
                # Define the 1-sigma bounds
                upper_bound = y_fit + sig_y
                lower_bound = y_fit - sig_y
                
                res_std = np.sqrt(output.res_var)

                # --- zero calculations ---
                avg_zero_sig = np.mean(zero_data[chan_col])
                stddev_zero_sig = np.std(zero_data[chan_col])
                zero_ppt = avg_zero_sig / slope  
                LOD = (3 * stddev_zero_sig) / slope
                    
        
                if plot:
                    # Changed to 1 row, 3 columns to include the Zero plot
                    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
                    fig.subplots_adjust(left=0.06, right=0.75, wspace=0.3)
                    fig.suptitle(f'NO-LIF LOD Calculation: {channel} (Cal {cal_num})', fontsize=14)
    
                    # --- 1. Cal Time Series Plot ---
                    axes[0].plot(cal_df[chan_col].values, color='lightgrey', alpha=0.3, label='Raw Cal')
                    axes[0].scatter(np.where(cal_df['stable_cal_point'])[0], 
                                    cal_df.loc[cal_df['stable_cal_point'], chan_col], 
                                    c='blue', s=3, label='Stable Points')
                    axes[0].set_title('Time Series Calibration')
                    axes[0].set_xlabel('Data Point Index')
                    axes[0].set_ylabel('Signal (counts s⁻¹ mW⁻¹)')
                    axes[0].legend(loc='upper right', fontsize=9)
    
                    # --- 2. Sensitivity Fit Plot ---
                    axes[1].scatter(X, Y, alpha=0.4, label='Data', s=15)
                    axes[1].plot(X, slope*X + intercept, 'r--', linewidth=2, label='ODR Fit')
                    axes[1].set_title('Calibration Sensitivity')
                    axes[1].set_xlabel(f'{molecule} Mixing Ratio (ppt)')
                    axes[1].set_ylabel('Signal (counts s⁻¹ mW⁻¹)')
                    axes[1].legend()
                    
                    # --- 3. Zero Signal Plot (The New Plot) ---
                    zero_vals = zero_data[chan_col].values
                    axes[2].plot(zero_vals, color='black', linewidth=0.8, alpha=0.7)
                    # Add a horizontal line for the Mean and the 3rd Standard Deviation
                    axes[2].axhline(avg_zero_sig, color='red', linestyle='-', label='Mean Zero')
                    axes[2].axhline(avg_zero_sig + (3 * stddev_zero_sig), color='orange', 
                                    linestyle='--', label='3σ Threshold')
                    
                    axes[2].set_title(f'Zero Task Signal ({zero_task})')
                    axes[2].set_xlabel('Zero Point Index')
                    axes[2].set_ylabel('Normalized Signal')
                    axes[2].legend(loc='upper right', fontsize=9)
                    
                    # --- Summary Text Box ---
                    LOD_text = (
                        f'SENSITIVITY RESULTS\n'
                        f'-------------------\n'
                        f'Intercept    = {intercept: .2f}\n'
                        f'Sensitivity  = {slope: .4f} cts/ppt\n\n'
                        f'ZERO STATS\n'
                        f'----------\n'
                        f'Avg Signal   = {avg_zero_sig: .4f}\n'
                        f'Std Dev (σ)  = {stddev_zero_sig: .4f}\n\n'
                        f'FINAL LOD\n'
                        f'---------\n'
                        f'3σ LOD       = {LOD: .2f} ppt'
                    )
                    
                    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
                    fig.text(0.77, 0.5, LOD_text, fontsize=10, family='monospace',
                             verticalalignment='center', horizontalalignment='left', bbox=props)
    
                    plt.show()      
# =============================================================================
#             if plot:
#                 fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#                 fig.subplots_adjust(left=0.08, right=0.70, wspace=0.25)
#                 fig.suptitle(f'NO-LIF LOD Calculation: {channel} (Cal {cal_num})', fontsize=14)
# 
#                 # Cal plot  
#                 axes[0].plot(cal_df[chan_col].values, color='lightgrey', alpha=0.3)
#                 axes[0].scatter(np.where(cal_df['stable_cal_point'])[0], 
#                                   cal_df.loc[cal_df['stable_cal_point'], chan_col], c='blue', s=3)
#                 axes[0].set_title('Time Series Cal')
#                 axes[0].set_xlabel('Data Point Index')
#                 axes[0].set_ylabel('Signal diff counts s-1 mW-1')
# 
#                 # Sensitivity Fit 
#                 axes[1].scatter(X, Y, alpha=0.4, label='Data', s=15)
#                 axes[1].plot(X, slope*X + intercept, 'r--', linewidth=2, label='ODR Fit')
#                 axes[1].set_title('Cal sensitivity')
#                 axes[1].set_xlabel(f'{molecule} Mixing Ratio (ppt)')
#                 axes[1].set_ylabel('Signal diff counts s-1 mW-1')
#                 axes[1].legend()
#                 
#                 # Summary Text Box
#                 LOD_text = (
#                     f'Intercept = {intercept: .2f} cts s-1 mW-1\n'
#                     f'Sensitivity         = {slope: .2f} cts s-1 mW-1 ppt-1\n'
#                     f'Avg signal (zero)   = {avg_zero_sig: .2f} cts s-1 mW-1\n'
#                     f'Calculated NO (zero)= {zero_ppt: .2f} ppt\n'
#                     f'3 sigma LOD         = {LOD: .2f} ppt'
#                 )
#                 
#                 props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
#                 
#                 fig.text(0.72, 0.5, LOD_text, fontsize=11, family='monospace',
#                          verticalalignment='center', horizontalalignment='left', bbox=props)
# 
#                 plt.show()
# =============================================================================
# =============================================================================
#            # --- PHASE D: DIAGNOSTICS ---
#             if plot:
#                 fig, axes = plt.subplots(1, 2, figsize=(16, 5))
#                 plt.tight_layout(rect=[0, 0, 0.75, 1])
#                 fig.suptitle(f'NO-LIF LOD Calculation: {channel} (Cal {cal_num})', fontsize=14)
#                 
#                  
# 
# 
#                 # 1. Stability Plot (Linear)
#                 axes[0].plot(cal_df[chan_col].values, color='lightgrey', alpha=0.5)
#                 axes[0].scatter(np.where(cal_df['stable_cal_point'])[0], 
#                                   cal_df.loc[cal_df['stable_cal_point'], chan_col], c='blue', s=5)
#                 axes[0].set_title('Time Series Stability')
#                 axes[0].set_xlabel('Data Point Index')
#                 axes[0].set_ylabel('Normalized Signal (cts/ref)')
# 
#                 # 2. Sensitivity Fit (Linear ODR)
#                 axes[1].scatter(X, Y, alpha=0.4, label='Data')
#                 axes[1].plot(X, slope*X + intercept, 'r--')
#                 axes[1].set_title('Sensitivity (ODR)')
#                 axes[1].set_xlabel(f'{molecule} Mixing Ratio')
#                 axes[1].set_ylabel('Normalized Signal (cts/ref)')
#                 axes[1].legend()
#                 
#                 LOD_text = (
#                     f'Background (zero NO) = {intercept: .2e} norm counts s-1\n'
#                     f'Sensitivity = {slope: .2e} norm counts s-1 ppt-1\n'
#                     f'Average signal during zero = {avg_zero_sig: .2e} norm counts s-1\n'
#                     f'Calculated NO during zero = {zero_ppt: .2f} ppt\n'
#                     f'3 sigma LOD from zero = {LOD: .2f} ppt'
#                     )
#                 
#                 props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
#                 fig.text(0.78, 0.5, LOD_text, fontsize=10, family='monospace',
#                          verticalalignment='center', horizontalalignment='left', bbox=props)
# 
#                 plt.show()
# =============================================================================


cts_data = cts_data.set_index('Date_time')
cts_data_resampled = cts_data.resample('1s').mean()
cts_data_resampled = cts_data_resampled.reset_index()
calc_LOD(cts_data_resampled, data_dir, channels, molecule, cal_cylinder_conc, plot, cal_task, zero_task)