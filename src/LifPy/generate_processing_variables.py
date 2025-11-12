import os
import re

import pandas as pd
import lif_functions as lif

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

data_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
               'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\Data2')
campaign_subdirs = os.listdir(data_dir)
day_folders = [folder for folder in campaign_subdirs 
                if re.match("20[0-9]{2}[0-1][0-9][0-3][0-9]", folder) is not None]
    
log_records = []
bin_records = []
# for day in all days
for day in day_folders:
    
    # Parse Log files
    log_dir = os.path.join(data_dir, day, f"LIFLog_{day}")
    # for logfile in all logfile
    for log_file in os.listdir(log_dir):
        if log_file[0:6] != 'LIFLog':  # Check first 6 characters, skip any non logfiles
            continue
        # Extract unformatted time from logfile
        with open(os.path.join(log_dir, log_file), "r") as infile:  # Read first line
            raw_time = infile.readlines()[0]
        # parse time
        time_split = re.match("Log File Created @ (\\d+):(\\d+):(\\d+)\\.\\d+\\s+(\\d+)/(\\d+)/(\\d+)", raw_time)
        time_groups = time_split.groups()
        time_fmt = f"{int(time_groups[4]):02}/{int(time_groups[3]):02}/20{time_groups[5]} {int(time_groups[0]):02}:{int(time_groups[1]):02}:{int(time_groups[2]):02}"
        # Save filename + time
        log_records.append({'log_filename': log_file, 'log_start_datetime': time_fmt})
        
    # Parse binary files
    bin_dir = os.path.join(data_dir, day, f"LIFCnts_{day}")
    for bin_file in os.listdir(bin_dir):
        if bin_file[0:7] != 'LIFCnts':  # Check first 6 characters, skip any non binfiles
            continue
        
        file_size = os.path.getsize(os.path.join(bin_dir, bin_file)) / 1024 / 1024
        bin_records.append({'date': day, 'bin_filename': bin_file, 'bin_size': file_size})

# Combine log metadata
log_df = pd.DataFrame.from_records(log_records)
log_df['dt'] = pd.to_datetime(log_df['log_start_datetime'], dayfirst=True)
log_df.sort_values('dt', inplace=True)
log_df['restart_index'] = range(log_df.shape[0])

# Combine bin metadata
bin_df = pd.DataFrame.from_records(bin_records)
bin_df.sort_values('bin_filename', inplace=True)

# Identify files that are from a restart as being the next file after a non-full file
bin_df['is_full'] = bin_df['bin_size'] >= 19.07
bin_df['is_restart'] = (~bin_df['is_full']).shift(1, fill_value=False)
bin_df['is_time_reset'] = False

for i in bin_df.index:
    if bin_df['is_restart'][i]:
        bin_data = lif.import_bin_data(data_dir, bin_df['date'][i]
                                   , bin_df['bin_filename'][i]
                                      )
        bin_data_dict = lif.deinterleave_bin_data(bin_data, channel_format
                                              , channel_count)
        bin_data_df = pd.DataFrame.from_dict(bin_data_dict)
        if bin_data_df['time_ms'][0] < 10000:
            bin_df.loc[i, 'is_time_reset'] = True
   
# Create log file index by treating is_restart as 0/1 integers and using 
# cumulative sum to group together files from between each restart
bin_df['restart_index'] = bin_df['is_restart'].cumsum()
bin_df['time_reset_index'] = bin_df['is_time_reset'].cumsum()

# creating mask for soft restarts 
soft_restart_mask = (bin_df['is_restart']) & (~bin_df['is_time_reset'])
soft_restart_index_list = bin_df['restart_index'][soft_restart_mask]

# generate processing variables df 
ignore_log_mask = ~log_df['restart_index'].isin(soft_restart_index_list)
log_df_processing_var = log_df[ignore_log_mask].copy()
log_df_processing_var['time_reset_index'] = range(log_df_processing_var.shape[0])
log_df_processing_var.drop(columns=['restart_index'], inplace=True)
processing_var_df = pd.merge(bin_df, log_df_processing_var, on='time_reset_index')
processing_var_df = processing_var_df[['date', 'bin_filename', 'log_filename', 'log_start_datetime', 'restart_index']]

# generate soft restarts df
soft_restart_filename_list = bin_df['bin_filename'][soft_restart_mask]
soft_restart_df = processing_var_df[processing_var_df['bin_filename'].isin(soft_restart_filename_list)].copy()
num_soft_restarts = len(soft_restart_df)

# generate to csv and
# print diagnostics to the console
processing_variables_file_path = os.path.join(data_dir, 'processing_variables.txt')
processing_var_df.to_csv(processing_variables_file_path, index=False)
print(f'Processing variables file created at:\n{processing_variables_file_path}'
      f'\n{num_soft_restarts} soft restarts found'
      )
if num_soft_restarts > 0:
    soft_restarts_file_path = os.path.join(data_dir, 'soft_restarts.txt')
    soft_restart_df.to_csv(soft_restarts_file_path, index=False)
    print(f'Soft restarts file created at: \n{soft_restarts_file_path}')
