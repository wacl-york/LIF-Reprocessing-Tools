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

def misaligned_counts(data_dir, day_folders, channels_to_use):
    
    HK_data = lif.import_HK_data(data_dir, day_folders)
    processing_variables = pd.read_csv(os.path.join(data_dir, 'processing_variables.txt'))
    soft_restarts = pd.read_csv(os.path.join(data_dir, 'soft_restarts.txt'))
    

    for i in soft_restarts.index:
        
        print(f"\nchecking for misaligned data in file: {soft_restarts['bin_filename'][i]}")
        
        log_start_datetime_seconds = lif.format_log_start_datetime(soft_restarts['log_start_datetime'][i])
        bin_data = lif.import_bin_data(data_dir, str(soft_restarts['date'][i]), str(soft_restarts['bin_filename'][i]))
        bin_data_dict = lif.deinterleave_bin_data(bin_data, channel_format, channel_count)
        bin_time_arr, HK_start_ind, HK_end_ind = lif.align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data)
        
        
        bin_time_ms = bin_data_dict['time_ms']
        tot_steps = len(bin_time_ms) - 1
        iter_range = iter(range(tot_steps))
        
        data_records = []
        
        for j in iter_range:
            
            if j % 100 == 0:
                print('\r%.2f' % (abs(1 - (tot_steps - j) / tot_steps) * 100)
                      , end='')
                
            curr_time = bin_time_arr[j]
            
            HK_ind = HK_start_ind + lif.find_min_ind(
                curr_time, HK_data['Time_s'], HK_start_ind, HK_end_ind
                )
            
            new_record = {}
            
            for channel in channels_to_use:
                new_record[channel] = bin_data_dict[channel][j]
                
            new_record['seed_LD_mode'] = bin_data_dict['seed_LD_mode'][j]
                
            new_record['Task'] = HK_data['Task'][HK_ind]
            
            data_records.append(new_record)
    
        test_data = pd.DataFrame(data_records)
        
    return test_data, bin_data_dict

def misaligned_counts_optimised(data_dir, day_folders, channels_to_use):
    
    HK_data = lif.import_HK_data(data_dir, day_folders)
    processing_variables = pd.read_csv(os.path.join(data_dir, 'processing_variables.txt'))
    soft_restarts = pd.read_csv(os.path.join(data_dir, 'soft_restarts.txt'))
    
    test_data_list = []
    bin_data_dict_list = []

    for i in soft_restarts.index:
        
        print(f"\nchecking for misaligned data in file: {soft_restarts['bin_filename'][i]}")
        
        log_start_datetime_seconds = lif.format_log_start_datetime(soft_restarts['log_start_datetime'][i])
        bin_data = lif.import_bin_data(data_dir, str(soft_restarts['date'][i]), str(soft_restarts['bin_filename'][i]))
        bin_data_dict = lif.deinterleave_bin_data(bin_data, channel_format, channel_count)
        bin_time_arr, HK_start_ind, HK_end_ind = lif.align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data)
        
        HK_time_slice = HK_data['Time_s'].values[HK_start_ind : HK_end_ind + 1]
        
        HK_ind_relative = np.searchsorted(HK_time_slice, bin_time_arr)
        HK_ind_relative = np.clip(HK_ind_relative, 0, len(HK_time_slice) - 1)
        HK_ind_absolute = HK_start_ind + HK_ind_relative
        new_df = pd.DataFrame()
        for channel in channels_to_use:
             # Ensure channel data is an array/list, and slice off the last element if tot_steps was len-1
            new_df[channel] = bin_data_dict[channel][:len(bin_time_arr)]
        new_df['seed_LD_mode'] = bin_data_dict['seed_LD_mode'][:len(bin_time_arr)]
        new_df['Task'] = HK_data['Task'].iloc[HK_ind_absolute].values
        test_data_list.append(new_df)
        bin_data_dict_list.append(bin_data_dict)

    if test_data_list:
        final_test_data = pd.concat(test_data_list, ignore_index=True)
    else:
        final_test_data = pd.DataFrame()        
        
    return final_test_data, bin_data_dict

        
test_data, bin_data_dict = misaligned_counts_optimised(data_dir, day_folders, channels_to_use = ["sig_A_counts", "sig_B_counts", "ref_counts", "seed_LD_current"])
            
                
                
            