import os

import numpy as np
import pandas as pd
import lif_functions as lif


data_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
               'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\Data2')

channel_count = 11  # 10 = single channel, 11 = dual channel

channel_format ={
    'sig_A_counts': 0
    , 'ref_counts': 1
    , 'seed_LD_current': 2
    , 'laser_pwr_PT0': [3, 4]
    , 'time_ms': [7, 8]
    , 'seed_LD_mode': 9
    , 'sig_B_counts': 10
}

day_folders = lif.find_day_folders(data_dir)

def misaligned_counts(data_dir, day_folders, channels_to_use, cal_task=2):

    HK_data = lif.import_HK_data(data_dir, day_folders)
    processing_variables_file_path = os.path.join(data_dir, 'processing_variables.txt')
    processing_variables = pd.read_csv(processing_variables_file_path).copy()
    soft_restarts = pd.read_csv(os.path.join(data_dir, 'soft_restarts.txt')).copy()
    
    for channel in channels_to_use:
        soft_restarts[f'{channel}_shift'] = 0
        

    # Loop through each soft restart 
    for i in soft_restarts.index:
        
        print(f"\rchecking for misaligned data in file: {soft_restarts['bin_filename'][i]}", end='')
        
        log_start_datetime_seconds = lif.format_log_start_datetime(soft_restarts['log_start_datetime'][i])
        bin_data = lif.import_bin_data(data_dir, str(soft_restarts['date'][i]), str(soft_restarts['bin_filename'][i]))
        bin_data_dict = lif.deinterleave_bin_data(bin_data, channel_format, channel_count)
        bin_time_arr, HK_start_ind, HK_end_ind = lif.align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data)
        
        # Vectorised alignment
        # Get the full 'Time_s' array from the HK data dictionary
        HK_time = HK_data['Time_s']
        
        # Slice the HK time array to only include the relevant segment (based on indices from lif.align_bin_HK).
        # Note: Since HK_data['Time_s'] is a NumPy array, we slice it directly.
        HK_time_slice = HK_time[HK_start_ind : HK_end_ind + 1]
        
        # Use np.searchsorted to find the index in HK_time_slice where each value in bin_time_arr 
        # would need to be inserted to maintain order. This quickly performs the time-alignment
        # for ALL Bin time points at once.
        HK_ind_relative = np.searchsorted(HK_time_slice, bin_time_arr)
        
        # Clamp the indices to the valid range (0 to length-1) within the HK_time_slice segment.
        HK_ind_relative = np.clip(HK_ind_relative, 0, len(HK_time_slice) - 1)
        
        # Convert the relative indices (within the slice) back to absolute indices 
        # for the full HK_data arrays.
        HK_ind_absolute = HK_start_ind + HK_ind_relative
        
        data_df = pd.DataFrame()
        
        # Pull the high-frequency Bin data columns (counts, etc.) directly using array slicing.
        for channel in channels_to_use:
             # Slices the array up to the length of the bin_time_arr
             data_df[channel] = bin_data_dict[channel][:len(bin_time_arr)]
             
        # Add the 'seed_LD_mode' column from the Bin data
        data_df['seed_LD_mode'] = bin_data_dict['seed_LD_mode'][:len(bin_time_arr)]
        
        # Vectorized Lookup: Use the array of absolute indices (HK_ind_absolute) to select 
        # the corresponding 'Task' status from the HK data in a single, fast operation.
        # Note: Since HK_data['Task'] is a NumPy array, we index it directly.
        data_df['Task'] = HK_data['Task'][HK_ind_absolute]
        
        test_section = data_df.loc[data_df['Task'] == cal_task].iloc[2200:4200]
        
        for channel in channels_to_use:
        # Container for sum of stds        
            sd_results = []
           
            # For each lag (0, 1, -1)
            #these are the defined shifts. These may change.
            for lag in [0, -1, 1]:
                #simply makes the value 0 and then we add onto this
                sum_of_sds = 0
           
                # For each mode
                #offline ==5
                #online ==6
                for mode in [5, 6]:
                   
                    # Get raw data
                    # using the above to do the shifts on the data by what seed ld mode we have and the
                    #channels specified from channel data
                    values = test_section.loc[test_section['seed_LD_mode'] == mode, channel]
                    values_lagged = values.shift(lag)
                    # Calculate mean & sd
                    #add it on to the 0 from the sum_of_stds and then also the new calulated values
                    sum_of_sds += values_lagged.std()
               
                # Save sum of sds for this lag
                sd_results.append((sum_of_sds, lag))
               
            # Find lag with minimum sum of sds. NB: min() of a list of tuples sorts by
            # first tuple entry, i.e. sd.
            best_lag = min(sd_results)[1]
            
            soft_restarts.loc[f'{channel}_shift'] = best_lag        
        

    # merge the shift values onto the processing variables df based on restart index
    shift_columns  = [f'{channel}_shift' for channel in channels_to_use]
    columns_to_merge = shift_columns + ['restart_index']

    processing_variables_shifts = pd.merge(
        processing_variables
        , soft_restarts[columns_to_merge]
        , on='restart_index'
        , how='left'               
    )
    
    processing_variables_shifts[shift_columns] = processing_variables_shifts[shift_columns].fillna(0)
    processing_variables_shifts = processing_variables_shifts.drop(columns=['restart_index'])

    ### Overwrite the processing variables file with the shifts appended
    processing_variables_shifts.to_csv(processing_variables_file_path, index=False)
    print('\nProcessing variables file updated to include shift values')

        
misaligned_counts(data_dir, day_folders, channels_to_use = ["sig_A_counts", "sig_B_counts", "ref_counts", "seed_LD_current"])
            
                
                
            