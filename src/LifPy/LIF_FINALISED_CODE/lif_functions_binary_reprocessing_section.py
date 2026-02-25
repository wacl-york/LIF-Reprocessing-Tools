# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:50:22 2026

@author: Eve
"""

import csv
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import islice
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.stats import linregress
from scipy import stats

from datetime import datetime as dt
import datetime
import pytz

from windrose import WindroseAxes


"""
This first section contains all of the sub-functions that are subsequently 
called from the reprocess_binary_data function. This generates processed data 
files containing on, off and diff counts data (both unnormalised for laser 
power and normalised for laser power) for each of the channels, as well as any 
HK data that has been specified.The function can be called with a data 
frequency of 10 (calculated 10Hz data using the online and offline points) or 
100 (showing all 100Hz data for visual inspection).

"""
def find_min_ind(target, array, start=0, end='full'):
    """
    Finds the index of the element in a segment of an array that is closest to 
    a target value.
    The index returned is relative to the start of the segmented array, not 
    the original array.
    This is a helper function that is called in various places.

    Parameters
    ----------
    target : float or int
        The numerical value to find the closest match for.
    array : array_like
        The array (list, NumPy array, etc.) to search through.
    start : int, optional
        The starting index for the search (inclusive). Default is 0.
    end : int or str, optional
        The ending index for the search (exclusive). If 'full', the search
        continues to the end of the array. Default is 'full'.

    Returns
    -------
    int
        The index within the *segmented* array that contains the value 
        closest to the target.
    """

    if end == 'full':
        array = array[start::]
    else:
        array = array[start: end]

    diff_arr = list(abs(np.array(array) - target))

    return diff_arr.index(np.min(diff_arr))

def find_day_folders(data_dir):
    """
    Identifies subdirectories within a given path that appear to be date 
    folders.
    
    A folder is considered a 'day folder' if its name matches the YYYYMMDD
    format, specifically starting with '20' for the year (e.g., 20240115).
    
    Parameters
    ----------
    data_dir : str
        The path to the main directory to search within.
    
    Returns
    -------
    list of str
        A list of strings, where each string is the name of a subdirectory
        that matches the YYYYMMDD date format.
    """
    
    campaign_subdirs = sorted(os.listdir(data_dir))
    day_folders = [folder for folder in campaign_subdirs 
                    if re.match("20[0-9]{2}[0-1][0-9][0-3][0-9]", folder) 
                    is not None]
    
    return day_folders

def gen_processing_var(data_dir, day_folders, channel_format, channel_count):
    """
    Generates and saves two metadata files, 'processing_variables.txt' and
    'soft_restarts.txt', which link binary count files (LIFCnts) to their
    corresponding log files (LIFLog) and identify specific system restart 
    events.
    
    This function performs the following main steps:
    1.  **Collects Metadata:** Iterates through 'day folders' to find all LIFLog
        and LIFCnts files, extracting log start times, binary file sizes, and
        creating initial records.
    2.  **Analyzes Restarts:** Determines if a binary file is likely a restart
        file based on the size of the *previous* file being non-full (less than
        19.07 MB).
    3.  **Identifies Time Resets (Hard Restarts):** For restart files, it loads
        the binary data and checks if the initial timestamp ('time_ms') is small
        (less than 10,000 ms), indicating a time counter reset (hard restart).
    4.  **Generates Indices:** Creates cumulative indices to group files between
        hard and soft restarts.
    5.  **Merges Data:** Combines the binary file metadata with the log file
        metadata, excluding log files that correspond only to 'soft restarts'
        (where the time counter did *not* reset).
    6.  **Saves Output:** Writes the final linked processing variables and the
        soft restart records to separate CSV files in the main data directory.
    
    Parameters
    ----------
    data_dir : str
        The root directory containing the daily data folders.
    day_folders : list of str
        A list of subdirectories (day folders, e.g., '20240115') to process.
    channel_format : str
        A string defining the data format of the binary channels.
    channel_count : int
        The expected number of data channels in the binary files.
    
    Returns
    -------
    None
        The function does not return a value but writes two files to disk:
        'processing_variables.txt' and 'soft_restarts.txt'.
        It also prints diagnostic information to the console.
    """
    
    print('Generating Processing Variables')
    
    log_records = []
    bin_records = []
    # for day in all days
    for day in day_folders:
        
        # Parse Log files
        log_dir = os.path.join(data_dir, day, f"LIFLog_{day}")
        # for logfile in all logfile
        # TODO check if need to sort output from os.listdir
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
        for bin_file in sorted(os.listdir(bin_dir)):
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
            bin_data = import_bin_data(data_dir, bin_df['date'][i]
                                       , bin_df['bin_filename'][i]
                                          )
            bin_data_dict = deinterleave_bin_data(bin_data, channel_format
                                                  , channel_count)
            #original version ,for when time is still positive
            #bin_data_df = pd.DataFrame.from_dict(bin_data_dict)
            #if bin_data_df['time_ms'][0] < 10000:
                #bin_df.loc[i, 'is_time_reset'] = True
            first_ms = bin_data_dict['time_ms'][0]
            if first_ms < 0:
                first_ms += 4294967296     
            if first_ms < 10000:
                bin_df.loc[i, 'is_time_reset'] = True
            #the above is for the TAS data only.
       
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
    print(f'\nProcessing variables file created at:\n{processing_variables_file_path}'
          f'\n\n{num_soft_restarts} soft restarts found'
          )
    if num_soft_restarts > 0:
        soft_restarts_file_path = os.path.join(data_dir, 'soft_restarts.txt')
        soft_restart_df.to_csv(soft_restarts_file_path, index=False)
        print(f'\nSoft restarts file created at: \n{soft_restarts_file_path}')
    
def import_HK_data(data_dir, day_folders):
    """
    Imports and concatenates Housekeeping (HK) data from all files across 
    multiple specified 'day folders'.
    
    The function iterates through each day folder, finds the LIFHK_{day} 
    subdirectory,reads all files starting with 'LIFHK' inside it, and 
    aggregates the data.
    All column data across all files is collected into a single list of arrays
    for each unique header, and then concatenated into a single NumPy array per
    header before returning.
    
    Parameters
    ----------
    data_dir : str
        The path to the root directory containing the daily data subdirectories.
    day_folders : list of str
        A list of subdirectory names (e.g., '20240115') to search for HK data.
    
    Returns
    -------
    dict
        A dictionary where keys are the column headers (from the input files),
        and values are 1D NumPy arrays containing the concatenated data
        from all processed files for that column.
    
    Notes
    -----
    - The function assumes HK files are space-delimited and contain a header row.
    - A warning is printed if an expected HK directory is not found for a day.
    - Data for each header is collected in a list of NumPy arrays and then
      concatenated once using 'np.concatenate' for efficiency.
    """
    
    print('\nreading HK files:\n')
    
    HK_data = {}
    
    for day in day_folders:
        
        file_list = []
        HK_dir = os.path.join(data_dir, day, f"LIFHK_{day}")
        if not os.path.isdir(HK_dir):
            print(f"Warning: Directory not found for day {day}: {HK_dir}")
            continue
        
        file_list.extend(file_name for file_name in os.listdir(HK_dir) 
                         if file_name.startswith('LIFHK'))

        if not file_list:
            continue

        for file in sorted(file_list):

            print(f'\r{file}', end='')
            
            file_path = os.path.join(HK_dir, file)
            file_data = pd.read_csv(file_path, delimiter=r'\s+', header=0)
    
            for header in list(file_data):
                
                current_data = file_data[header].values
                
                if header in HK_data:
                    HK_data[header].append(current_data)
                else:
                    # Initialize with a list containing the first array
                    HK_data[header] = [current_data]
    
    final_HK_data = {}
    for header, list_of_arrays in HK_data.items():
        # Perform one single concatenation for each header
        final_HK_data[header] = np.concatenate(list_of_arrays) 
    print('\nHK data imported successfully')           

    return final_HK_data

def gen_bin_file_list(bin_file_path, skip_start_bin, skip_end_bin):
    """
    Generates a list of files from a directory, excluding specified files 
    from the beginning and end of the list.

    The function first lists all files in the given path and then uses 
    slicing to exclude files based on the skip parameters.

    Parameters
    ----------
    bin_file_path : str
        The path to the directory containing the files.
    skip_start_bin : int
        The number of files to skip (discard) from the beginning of the list.
    skip_end_bin : int
        The number of files to skip (discard) from the end of the list.

    Returns
    -------
    list of str
        A list containing the names of the remaining files after skipping 
        the specified files from the start and end.

    Raises
    ------
    Exception
        Catches any general error during the list slicing, printing a 
        specific message if the skip parameters are set to discard all files.
    """
    
    # TODO check if need to sort the os.listdir output
    file_list = [f for f in os.listdir(bin_file_path) \
                 if os.path.isfile(os.path.join(bin_file_path, f))]
    try:
        file_list = file_list[skip_start_bin: len(file_list) - skip_end_bin]
    except Exception as err:
        if skip_end_bin + skip_start_bin >= len(file_list):
            print('Error - skip_start and skip_end are set to discard all \
                  binary files.')
        else:
            print(err)
            
    return file_list

def format_log_start_datetime(log_start_datetime):
    """
    Converts a date/time string into an 'epoch time' measured in seconds 
    elapsed since 01/01/1904.
    The function attempts conversion using two common date/time formats and 
    includes logic to strip milliseconds if conversion fails due to unconverted 
    data remaining.

    Parameters
    ----------
    log_start_datetime : str
        The date/time string to be converted. Expected primary format is 
        'DD/MM/YYYY HH:MM:SS'.

    Returns
    -------
    float
        The number of seconds elapsed between '01/01/1904 00:00:00' and 
        the input timestamp.

    Raises
    ------
    SystemExit
        Exits the program if all conversion attempts fail, printing an error 
        message.

    Notes
    -----
    The function handles three common datetime string issues:
    1. Primary format: 'DD/MM/YYYY HH:MM:SS'
    2. Alternative format: 'HH:MM:SS DD/MM/YYYY'
    3. Trailing milliseconds, which are stripped before a final conversion 
    attempt.
    
    """
    
    try:
         epoch_time = (dt.strptime
                       (log_start_datetime, '%d/%m/%Y %H:%M:%S') -
                       dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()
    except Exception as err:
        print('\n' + str(err))
        if 'does not match format' in str(err):
            print('\nTrying alternative timestamp formatter')
            try:
                epoch_time = (dt.strptime
                              (log_start_datetime,'%H:%M:%S %d/%m/%Y') -
                              dt.strptime
                              ('01/01/1904', '%d/%m/%Y')).total_seconds()
                print('Successfully coerced timestamp')
            except:
                print('\nUnable to convert log_start_datetime string, \
                      please check the input format. Exit - 01')
                sys.exit(1)

        if 'unconverted data remains:' in str(err):
            print('\nDeleting milliseconds from timestamp')
            try:
                epoch_time = (dt.strptime
                              (log_start_datetime.replace
                               (str(err).split(': ')[1], ''), \
                                   '%d/%m/%Y %H:%M:%S')
                               - dt.strptime
                               ('01/01/1904', '%d/%m/%Y')).total_seconds()
                print('Successfully coerced timestamp')
            except:
                print('\nUnable to convert log_start_datetime string, \
                      please check the input format. Exit - 01')
                sys.exit(1)
    
    return epoch_time

def import_bin_data(data_dir, date, file):
    """
    Imports data from a single binary file into a NumPy array.

    Parameters
    ----------
    bin_file_path : str
        The path to the directory containing the binary file.
    file : str
        The name of the binary file to be imported.

    Returns
    -------
    numpy.ndarray
        A 1-dimensional NumPy array containing the imported binary data.

    Notes
    -----
    The data type string 'dt='>i2'' specifies:
    - '>' : Big-endian byte order.
    - 'i' : Signed integer type.
    - '2' : 2 bytes (16 bits) in size.
    """
    
    bin_data_array = np.fromfile(os.path.join(data_dir, date, f'LIFCnts_{date}', file)
                           , dtype='>i2') 
    
    return bin_data_array

def deinterleave_bin_data(bin_data, channel_format, channel_count,
                           file, data_dir, rep_rate_Hz=200000):    
    
    frames = np.array(bin_data)
    decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

    bin_data_dict = {}

    for dict_key in channel_format.items():
        bin_data_dict[dict_key[0]] = []

    for channel, channel_ID in channel_format.items():
        
        if type(0) == type(channel_ID):
            bin_data_dict[channel] = np.concatenate(
                (bin_data_dict[channel], decimate_arr[channel_ID]))
            
        if type([]) == type(channel_ID):
            hi = decimate_arr[channel_ID[0]] * 65536
            lo = np.where(decimate_arr[channel_ID[1]] < 0
                          , decimate_arr[channel_ID[1]] + 65536
                          , decimate_arr[channel_ID[1]])
            bin_data_dict[channel] = np.concatenate(
                (bin_data_dict[channel], lo + hi))

    bin_data_dict['laser_pwr_PT0'] = bin_data_dict['laser_pwr_PT0'] / 100000

    max_cts = rep_rate_Hz / 100

#testing a new function format!
    processing_variables = pd.read_csv(os.path.join(
         data_dir, 'processing_variables.txt')
         )
    this_file_index = processing_variables[
         processing_variables['bin_filename'] == file].index[0]

    for channel in bin_data_dict.keys():
        bin_data_dict[channel] = bin_data_dict[channel].astype(float)
    for col in processing_variables.columns:
        if '_shift' in col:
            # Get the actual channel name (e.g., 'sig_counts' from 'sig_counts_shift')
            for suffix in ['']:
                channel_name = col.replace('_shift' , suffix)
            # Check if this channel exists in our data
            
                if channel_name in bin_data_dict:
                    shift_val = int(float(processing_variables[col][this_file_index]))
                    
                    if shift_val == 0:
                        continue
                        
                    data = bin_data_dict[channel_name]
                    orig_len = len(data)
                    
                    if shift_val > 0:
                        # Move Right: Pad start, trim end
                        shifted = np.pad(data, (shift_val, 0), mode='constant', constant_values=np.nan)
                        bin_data_dict[channel_name] = shifted[:orig_len]
                    else:
                        # Move Left: Pad end, trim start
                        abs_s = abs(shift_val)
                        shifted = np.pad(data, (0, abs_s), mode='constant', constant_values=np.nan)
                        bin_data_dict[channel_name] = shifted[abs_s:]
                    
                    print(f"Applied shift of {shift_val} to {channel_name}")
    for channel in list(channel_format):
        if 'counts' in channel:
            bin_data_dict[channel] = \
                np.array([cts if cts < max_cts else np.nan for cts \
                          in bin_data_dict[channel]])
    
            bin_data_dict[channel + '_lin'] = \
                -np.log(1 - (bin_data_dict[channel] / max_cts)) * max_cts
    
            bin_data_dict[channel + '_norm'] = \
                bin_data_dict[channel + '_lin'] \
                    / (bin_data_dict['laser_pwr_PT0'])
    #added this in to try and see if we can get to the positive time values! 21/01/2026
    #if 'time_ms' in bin_data_dict:
       # bin_data_dict['time_ms'] = bin_data_dict['time_ms'].astype(np.int64) % 4294967296
    return bin_data_dict
 
def gen_output_file(data_dir, data_freq, file, channel_format, HK_headers_dict
                    , HK_data, date, bin_data_dict):
   """
    Creates and initializes a new output text file for processed data, writes
    metadata, and generates the full data header row and a placeholder string
    for non-data periods.
    
    The function determines the next available file index for the current day,
    creates the output directory if necessary, opens the new file, writes a
    standard metadata block (including a timestamp), and then generates and
    writes the full combined header string based on the processing frequency.
    
    Parameters
    ----------
    data_dir : str
        The root directory of the project, used to construct the output folder 
        path.
    data_freq : int
        The data processing frequency (10 or 100 Hz), which dictates the
        structure and content of the output headers.
    file : str
        The name of the current binary input file (used for naming the output
        file).
    channel_format : dict
        A dictionary defining the raw data channels, used to generate headers
        for Signal Counts and other processed channels (10 Hz mode).
    HK_headers_dict : dict
        A dictionary of desired Housekeeping headers. This dictionary is
        checked against HK_data and modified in-place to remove non-existent
        headers.
    HK_data : dict
        The dictionary containing all Housekeeping data (used to validate that
        requested headers in HK_headers_dict actually exist).
    date : str
        The current day in YYYYMMDD format, used for naming the output 
        directory and file.
    bin_data_dict : dict
        The dictionary containing 100 Hz binary data channels. Used in 100 Hz
        mode to dynamically generate the list of binary data headers.
    
    Returns
    -------
    tuple
        (processed_file, nan_data)
        processed_file : file object
            The open file handle for the initialized output text file.
        nan_data : str
            A comma-separated string of placeholder values (-9999) matching the
            width of all data columns (excluding the first column, which is
            assumed to be the time column), used to fill periods of missing or
            invalid data.
    """
   
   output_dir = os.path.join(data_dir, date, f'LIFProcessed_{date}')
   try:
     os.makedirs(output_dir, exist_ok=True)
   except OSError as e:
     print(f"Error creating directory {output_dir}: {e}")
   
   # TODO check if need to order os.listdir output
   file_ind = str(len(os.listdir(output_dir))).zfill(2)
   
   output_filename = os.path.join(
        output_dir, 
        '%s_LIF_processed_data_%s.txt' % (date, file_ind)
    )
   
   processed_file = open(output_filename, 'w+') 
   
   met_add = '\nReprocessed on: ' + dt.strftime(dt.now(), '%Y/%m/%d %H:%M:%S')\
             + '\nTime reference: (seconds since 1904-01-01 00:00:00)' 
             
   cts_met = '\n\n' + open(
                       os.getcwd() + '\\lib\\lif_metadata.txt').read() + met_add
   
   processed_file.write(str(cts_met + '\n' + '\n'))
   
   if data_freq == 10:
   
       gen_headers = ('mac_time_s,lsr_pwr_on_mW,lsr_pwr_off_mW,lsr_pwr_mW,'
                   'ref_on_cts,ref_off_cts,ref_diff_cts,ref_on_cts_norm'
                   ',ref_off_cts_norm,ref_diff_cts_norm'
                   )
                           
       cts_labels = [name.split('_')[0] + ('_' + name.split('_')[1] 
                                           if len(name.split('_')) > 2 else '')
                     for name in list(channel_format)
                     if '_counts' in name and not name.startswith('ref_')
                     ]
       
       cts_headers = ','.join(['%s_on_cts,%s_off_cts,%s_diff_cts'
                               % (label, label, label) for label in cts_labels]) 
           
       cts_headers_norm = ','.join(['%s_on_cts_norm,%s_off_cts_norm,'
                                    '%s_diff_cts_norm' % (label, label, label)  
                                    for label in cts_labels])
            
       binary_headers = gen_headers + ',' + cts_headers + ',' + cts_headers_norm 
       
       for HK_ID in list(HK_headers_dict):
           try:
               HK_data[HK_ID]
           except Exception as err:
               print('\nThere is no HK data header called ' 
                     + str(err).replace("'", '') + ', so it will be discarded')
               del HK_headers_dict[HK_ID]
               
       HK_labels = ','.join(list(HK_headers_dict)) 
       
       all_headers = binary_headers + ',' + HK_labels + '\n'
          
   elif data_freq == 100:
       
       bin_headers = ','.join(list(bin_data_dict.keys()))
       HK_labels = ','.join(list(HK_headers_dict))
       all_headers = bin_headers + ',' + HK_labels + '\n'
       
   processed_file.write(all_headers)
   
   nan_data = ','.join(np.full(len(all_headers.split(',')[1::]), str(-9999)))
   
   return processed_file, nan_data

def align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data):
    """
    Aligns binary data timestamps with Housekeeping (HK) data times and 
    determines the corresponding HK indices.
    This function calculates absolute timestamps for the binned data and then 
    finds the starting and ending indices in the HK data that correspond to the 
    first and last binned data times.

    Parameters
    ----------
    bin_data_dict : dict
        A dictionary containing the binary data. 
    log_start_datetime_seconds : float or int
        The absolute start time of the log, in seconds(s).
    HK_data : dict
        A dictionary containing the HK data.

    Returns
    -------
    bin_time_arr : numpy.ndarray
        A 1D array of absolute timestamps for the binary data, in seconds (s).
    HK_start_ind : int
        The index in HK_data['Time_s'] that is closest to or immediately 
        before bin_time_arr[0].
    HK_end_ind : int
        The index in HK_data['Time_s'] that is closest to or immediately 
        before bin_time_arr[-1].

    Raises
    ------
    SystemExit
        If the calculated HK start index and HK end index are the same, 
        suggesting that the log_start_datetime_seconds might be incorrect.

    Notes
    -----
    This function relies on an external function, 
    find_min_ind(target_time, time_array), which is assumed to return the 
    index of the element in time_array closest to or immediately preceding 
    target_time.
    """
    # 1. Get the raw time array
    raw_ms = bin_data_dict['time_ms']
    
    # 2. Get the first millisecond value of this specific file
    # This is the 'zero' marker for this file.
    first_ms_in_file = raw_ms[0]
    
    # 3. Calculate time relative to the start of the file, then add to log start.
    # This cancels out the 3.8 billion offset.
    bin_time_arr = \
        ((raw_ms - first_ms_in_file) / 1000) + log_start_datetime_seconds
        
    HK_start_ind = find_min_ind(bin_time_arr[0], HK_data['Time_s']) 
    
    HK_end_ind = find_min_ind(bin_time_arr[-1], HK_data['Time_s'])
    
    if HK_start_ind == HK_end_ind:
        print('\n--- Alignment Failure ---')
        print(f"Log Start: {log_start_datetime_seconds}")
        print(f"Binary File First MS: {first_ms_in_file}")
        print(f"Calculated Absolute Time: {bin_time_arr[0]}")
        print(f"HK Time Range: {HK_data['Time_s'][0]} to {HK_data['Time_s'][-1]}")
        print('The log_start_datetime does not overlap with HK data.')
        sys.exit(1)
        
    return bin_time_arr, HK_start_ind, HK_end_ind
    """
    bin_time_arr = \
        (bin_data_dict['time_ms'] / 1000) + log_start_datetime_seconds
        
    HK_start_ind = find_min_ind(bin_time_arr[0], HK_data['Time_s']) 
    
    HK_end_ind = find_min_ind(bin_time_arr[-1], HK_data['Time_s'])
    
    if HK_start_ind == HK_end_ind:
        print('\nThe log_start_datetime is incorrect')
        print(bin_time_arr[0])
        sys.exit(1)
        
    return bin_time_arr, HK_start_ind, HK_end_ind 
    """
def gen_output_data(file, channel_format, data_dir, bin_data_dict, HK_data 
                    , HK_headers_dict, data_freq, bin_time_arr, HK_start_ind
                    , HK_end_ind, nan_data, processed_file, histograms):
    """
    Processes, averages, aligns, and writes binary and Housekeeping (HK) data 
    to the output file, supporting both 10Hz averaged data and 100Hz 
    instantaneous data.

    The function iterates through the 100Hz binary data. It handles periods of 
    constant laser operation (Mode 1) by writing NaN placeholders and applies 
    different processing logic based on the desired output frequency.

    Parameters
    ----------
    channel_format : dict
        Mapping of channel names to indices, used for identifying count 
        channels.
    bin_data_dict : dict
        Dictionary of processed (de-interleaved, corrected) binary data arrays 
        (100 Hz).
    HK_data : dict
        Housekeeping data dictionary.
    HK_headers_dict : dict
        Validated list of HK headers for writing.
    data_freq : int
        The intended output frequency (10 for averaged, 100 for instantaneous).
    bin_time_arr : numpy.ndarray
        Array of absolute timestamps for the binary data.
    HK_start_ind : int
        Starting index for relevant HK data.
    HK_end_ind : int
        Ending index for relevant HK data.
    nan_data : str
        Comma-separated string of placeholder values (-9999) for invalid data.
    processed_file : file object
        The open file handle for the output text file.
    shift_correction : int
        The index shift to apply to the 'seed_LD_mode' channel.
    cts_ind_arr : list of int
        Indices for linearized signal count channels.
    cts_ind_arr_norm : list of int
        Indices for normalized signal count channels.

    Returns
    -------
    None
        Data is written directly to the processed_file (side effect).
    """
    if data_freq == 10:
        seed_LD_mode = bin_data_dict['seed_LD_mode']
        if 5 not in seed_LD_mode and 6 not in seed_LD_mode:
            print('No periods of laser dither detected, no 10Hz data processed')
            return
    
    cts_ind_arr = [i for i in range(0, len(list(bin_data_dict)))
                       if 'sig_' in list(bin_data_dict)[i]
                           and '_lin' in list(bin_data_dict)[i]]
    
    cts_ind_arr_norm = [i for i in range(0, len(list(bin_data_dict))) 
                            if 'sig_' in list(bin_data_dict)[i]
                                and '_norm' in list(bin_data_dict)[i]]

    #pre fetching arrays for faster access in the loop
    bin_time_ms = bin_data_dict['time_ms']
    seed_LD_mode = bin_data_dict['seed_LD_mode']
    ref_counts_norm = bin_data_dict['ref_counts_norm']
    ref_counts = bin_data_dict['ref_counts']
    laser_pwr_PT0 = bin_data_dict['laser_pwr_PT0']
    HK_Time_s = HK_data['Time_s']
    HK_headers = list(HK_headers_dict)
    bin_keys = list(bin_data_dict.keys())
    cts_channel_data = [bin_data_dict[bin_keys[l]] for l in cts_ind_arr]
    cts_norm_channel_data = [bin_data_dict[bin_keys[l]] for l in cts_ind_arr_norm]
    
    bin_time = [0] 
    tot_steps = len(bin_time_ms) - 1
    iter_range = iter(range(tot_steps))
    output_lines = []
    histogram_on = []
    histogram_off = []
    no_dropped_online = 0
    one_dropped_online = 0
    two_dropped_online = 0
    processed_points = 0

    for i in iter_range: 
        
        # displays the progress percentage       
        if i % 1000 == 0:
            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100)
                  , end='') 
        
        curr_time = bin_time_arr[i]
        

        # finds the periods where the laser is in constant mode and
        # replaces all values with nan values
        # In seed LD mode, 1 defines constant mode
        if seed_LD_mode[i] != 1 and \
        seed_LD_mode[i + 1] == 1:
            for j in range(tot_steps - i):
                if seed_LD_mode[i + j] == 1 \
                and seed_LD_mode \
                    [i + j + 1] != 1:  
                    time_fill = \
                        list(np.arange(bin_time_ms[i]
                                       , bin_time_ms[i + j] + 1
                                       , 100))
                    bin_time = bin_time + time_fill  
                    for k in range(int(j / data_freq) + 1):
                        output_lines.append(
                        f"{curr_time + (1 / data_freq) * k},{nan_data}\n"
                        )
                    next(islice(iter_range, j, i), None)
                    break

        if data_freq == 10:

            # finds point of online/offline switch
            # In seed LD mode, 5 defines offline point, 6 defines online point
            if seed_LD_mode[i] == 6 and \
            seed_LD_mode[i + 1] == 5: 
                if (bin_time_ms[i] - bin_time[-1] != 0) and (i >= 7): 
                    bin_time.append(bin_time_ms[i]) 
                    
                    cts_wrt_list = []
                    cts_wrt_list_norm = []
                    
                    processed_points += 1
                    
                    if ref_counts_norm[i - 7] < \
                        np.mean(ref_counts_norm[i - 6: i + 1]) \
                        - 3 * np.std(ref_counts_norm[i - 6: i + 1]) \
                        or ref_counts_norm[i - 7] > \
                        np.mean(ref_counts_norm[i - 6: i + 1]) \
                        + 3 * np.std(ref_counts_norm[i - 6: i + 1]):
                            if ref_counts_norm[i - 6] < \
                                np.mean(ref_counts_norm[i - 5: i + 1]) \
                                - 3 * np.std(ref_counts_norm[i - 5: i + 1]) \
                                or ref_counts_norm[i - 6] > \
                                np.mean(ref_counts_norm[i - 5: i + 1]) \
                                + 3 * np.std(ref_counts_norm[i - 5: i + 1]):
                                    online_offset = 5   # removes  first 2 online points
                                    two_dropped_online += 1
                            else: 
                                online_offset = 6 # removes first online point
                                one_dropped_online += 1
                    else:
                            online_offset = 7  # keeps all online points  
                            no_dropped_online += 1                              
                        
                    
    
                    on_cts_ref = np.mean(ref_counts[i - online_offset: i + 1]
                        ) * 10 * data_freq
                    off_cts_ref = np.mean(ref_counts[i + 1: i + 3]
                        ) * 10 * data_freq
                    cts_diff_ref = on_cts_ref - off_cts_ref
                    
                    on_cts_ref_norm = np.mean(ref_counts_norm[i - online_offset: i + 1]
                        ) * 10 * data_freq
                    off_cts_ref_norm = np.mean(ref_counts_norm[i + 1: i + 3]
                        ) * 10 * data_freq
                    cts_diff_ref_norm = on_cts_ref_norm - off_cts_ref_norm
                    
                    histogram_on.append(on_cts_ref_norm)
                    histogram_off.append(off_cts_ref_norm)
    
    
                    for channel_data in cts_channel_data:
                        on_cts = np.mean(
                            channel_data[i - online_offset: i + 1]
                            ) * 10 * data_freq
                        off_cts = np.mean(
                            channel_data[i + 1: i + 3]
                            ) * 10 * data_freq
                        cts_diff = on_cts - off_cts
                        cts_wrt_list.append(
                            f'{on_cts},{off_cts},{cts_diff}'
                            )
    
                    for channel_data in cts_norm_channel_data:
                        on_cts_norm = np.mean(
                            channel_data[i - online_offset: i + 1]
                            ) * 10 * data_freq
                        off_cts_norm = np.mean(
                            channel_data[i + 1: i + 3]
                            ) * 10 * data_freq
                        cts_diff_norm = on_cts_norm - off_cts_norm
                        cts_wrt_list_norm.append(
                            f'{on_cts_norm},{off_cts_norm},{cts_diff_norm}'
                            )
                    
                    on_lsr_pwr = np.mean(
                        laser_pwr_PT0[i - online_offset: i + 1]
                        )
                    off_lsr_pwr = np.mean(
                        laser_pwr_PT0[i + 1: i + 3]
                        )
                    lsr_pwr = np.mean([on_lsr_pwr, off_lsr_pwr])
    
                    HK_ind = HK_start_ind + find_min_ind(
                        curr_time, HK_Time_s, HK_start_ind, HK_end_ind
                        )
    
                    if not math.isnan(cts_diff):
    
                        gen_write = f'{curr_time},{on_lsr_pwr},{off_lsr_pwr},{lsr_pwr}'
                        cts_write = f'{on_cts_ref},{off_cts_ref},{cts_diff_ref}' \
                                    f',{on_cts_ref_norm},{off_cts_ref_norm},{cts_diff_ref_norm}' \
                                    f',{",".join(cts_wrt_list)},{",".join(cts_wrt_list_norm)}'
                        HK_write = ','.join(
                            [str(HK_data[key][HK_ind]) for key in HK_headers]
                            )
    
                        output_lines.append(
                                f'{gen_write},{cts_write},{HK_write}\n'
                                )

        elif data_freq == 100:
            
            HK_ind = HK_start_ind + find_min_ind(
                curr_time, HK_Time_s, HK_start_ind, HK_end_ind
                )
         
            bin_values_at_i = [bin_data_dict[name][i] \
                               for name in bin_keys]
            bin_write = ','.join(map(str, bin_values_at_i))
            HK_write = ','.join(
                [f'{HK_data[key][HK_ind]}' for key in HK_headers]
            )
       
            output_lines.append(f'{bin_write},{HK_write}\n')   
         
    processed_file.write(''.join(output_lines))     

    if data_freq == 10:
        if processed_points > 0:
            
            print(f'\npercentage of no dropped online points = \
                  {((no_dropped_online/processed_points)*100): .2f}'
                  f'\npercentage of one dropped online point = \
                  {((one_dropped_online/processed_points)*100): .2f}'
                  f'\npercentage of two dropped online points = \
                  {((two_dropped_online/processed_points)*100): .2f}'
                  )
            
        else:
            print('\nNo processed points in file')
    
    if histograms:
        
        histogram_off_array = np.array(histogram_off)
        histogram_off_array = histogram_off_array[np.isfinite(histogram_off_array)]
        histogram_on_array = np.array(histogram_on)
        histogram_on_array = histogram_on_array[np.isfinite(histogram_on_array)]
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        
        ax1.hist(histogram_on_array, bins=100, color='blue')
        ax1.set_title('ref on cts (laser power normalised)')
        ax1.set_xlabel('counts')
        ax1.set_ylabel('Frequency')
        
        ax2.hist(histogram_off_array, bins=100, color='red')
        ax2.set_title('ref off cts (laser power normalised)')
        ax2.set_xlabel('counts')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def reprocess_binary_data(date, file, log_start_datetime, HK_headers_dict
                          , channel_format, data_dir, HK_data_dict, data_freq=10
                          , channel_count=10, histograms=False):
    """
    Orchestrates the complete processing of a single binary data file (LIFCnts)
    and its associated Housekeeping (HK) and Log data into a final, readable
    CSV format.
    
    This function acts as the main wrapper, calling a series of sub-functions
    to perform the following sequential steps:
    1.  **Format Time:** Converts the log file's start time string into a
        reference time in seconds.
    2.  **Import Data:** Reads the raw binary file data and deinterleaves it
        into a structured dictionary by channel.
    3.  **Initialize Output:** Creates and writes metadata and headers to the
        new output file.
    4.  **Align Data:** Calculates time shifts and identifies the corresponding
        start and end indices in the HK data that match the binary data time range.
    5.  **Write Data:** Processes the data (calculates counts, linearizes,
        normalizes, merges with HK data) and writes the output line-by-line.
    
    Parameters
    ----------
    date : str
        The date associated with the data being processed in the form 'YYYYMMDD'.
    file : str
        The file name of the specific binary data file (LIFCnts) being processed.
    log_start_datetime : str
        The start time string extracted from the corresponding LIFLog file.
    HK_headers_dict : dict
        A dictionary containing the desired Housekeeping data headers/channels.
        (This dictionary is modified in place during output file generation).
    channel_format : dict
        A dictionary defining the format and names of the data channels
        within the binary files (used for deinterleaving and header creation).
    data_dir : str
        The base directory where the daily data folders and output files are located.
    HK_data : dict
        The consolidated dictionary containing all available Housekeeping data
        read from all files in the current run.
    data_freq : int, optional
        The frequency of data to be returned (10 or 100 Hz). The default is 10.
    channel_count : int, optional
        The total number of channels in the binary data. The default is 10.
    histograms : bool, optional
        Flag to indicate whether histograms should be generated during
        processing (passed to the final data writing step). The default is False.
    
    Returns
    -------
    None
        The function performs file operations (reads data and writes a processed
        output file) but does not explicitly return a value.
    """
    
    print('\nReprocessing file %s' % file)
    
    log_start_datetime_seconds = format_log_start_datetime(
        log_start_datetime
        )
    bin_data = import_bin_data(
        data_dir, date, file
        )
    bin_data_dict = deinterleave_bin_data(
        bin_data, channel_format, channel_count, file, data_dir
        )
    processed_file, nan_data = gen_output_file(
        data_dir, data_freq, file, channel_format, HK_headers_dict
        , HK_data_dict, date, bin_data_dict
        )
    bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(
        bin_data_dict, log_start_datetime_seconds, HK_data_dict
        )
    gen_output_data(
        file, channel_format, data_dir, bin_data_dict, HK_data_dict, HK_headers_dict, data_freq
        , bin_time_arr, HK_start_ind, HK_end_ind, nan_data, processed_file
        , histograms
        )

def misaligned_counts(data_dir, day_folders, channel_format, 
                      channel_count, molecule, cal_task=2, plot=False):
    """
    Checks for and corrects misalignment (time lag) between binary data 
    channels.
    
    This check is only performed for data files immediately following a 'soft
    restart' (a system restart where the internal time counter did not reset).
    The function determines the optimal channel shift by finding the lag (0, -1,
    or +1 time step) that minimizes the standard deviation (SD) of the count
    data during a specific high-concentration calibration period (where SD 
    should be lowest).
    
    The results (shifts) are appended to the 'processing_variables.txt' file.
    
    Parameters
    ----------
    data_dir : str
        The root directory containing the daily data folders and the metadata 
        files.
    day_folders : list of str
        A list of subdirectory names (e.g., '20240115') containing the HK data.
    channel_format : dict
        A dictionary defining the format and names of all data channels in the
        binary files.
    channel_count : int
        The total number of channels in the binary data.
    molecule : str
        The name of the target molecule (e.g., 'H2O') used to identify the
        relevant HK calibration gas flow channel (e.g., 'Cal_H2O_MFC_set').
    cal_task : int, optional
        The integer value of the 'Task' Housekeeping channel that identifies a
        calibration run. Default is 2.
    plot : bool, optional
        If True, generates and displays plots showing the original vs. shifted
        data for visual inspection of the alignment. Default is False.
    
    Returns
    -------
    None
        The function performs file operations, printing diagnostics to the 
        console, and updates the 'processing_variables.txt' file with the 
        determined time shifts.
    """
    
    excluded_channels = ['laser_pwr_PT0', 'seed_LD_mode', 'time_ms']
    channels_to_use = [
        key for key in channel_format.keys() if key not in excluded_channels
    ]
    
    processing_variables_file_path = os.path.join(
        data_dir, 'processing_variables.txt'
    )
    processing_variables = pd.read_csv(
        processing_variables_file_path
    ).copy()
    
    soft_restarts_file_path = os.path.join(data_dir, 'soft_restarts.txt')
    if not os.path.exists(soft_restarts_file_path):
        print('\nNo soft restarts identified - misalignment check not required')
        return
        
    soft_restarts = pd.read_csv(soft_restarts_file_path).copy()
    HK_data = import_HK_data(data_dir, day_folders)
    
    for channel in channels_to_use:
        soft_restarts[f'{channel}_shift'] = 0
        
    unchecked_restarts = []

    # Loop through each soft restart 
    for i in soft_restarts.index:
        
        print(f"\nchecking for misaligned data in file: "
              f"{soft_restarts['bin_filename'][i]}" 
              f"(restart index: {soft_restarts['restart_index'][i]})"
              )
        
        log_start_datetime_seconds = format_log_start_datetime(
            soft_restarts['log_start_datetime'][i]
        )
        
        date_str = str(soft_restarts['date'][i])
        bin_filename = str(soft_restarts['bin_filename'][i])
        
        bin_data = import_bin_data(data_dir, date_str, bin_filename)
        bin_data_dict = deinterleave_bin_data(
            bin_data, channel_format, channel_count
        )
        bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(
            bin_data_dict, log_start_datetime_seconds, HK_data
        )
        
        # Vectorised alignment
        HK_time = HK_data['Time_s']
        
        # Slice the HK time array to only include the relevant segment
        HK_time_slice = HK_time[HK_start_ind : HK_end_ind + 1]
        
        # Find the index in HK_time_slice for each bin_time_arr value
        HK_ind_relative = np.searchsorted(HK_time_slice, bin_time_arr)
        
        # Clamp the indices to the valid range (0 to length-1)
        HK_ind_relative = np.clip(
            HK_ind_relative, 0, len(HK_time_slice) - 1
        )
        
        # Convert the relative indices back to absolute indices 
        HK_ind_absolute = HK_start_ind + HK_ind_relative
        
        data_df = pd.DataFrame()
        
        # Pull the high-frequency Bin data columns
        arr_len = len(bin_time_arr)
        for channel in channels_to_use:
            # Slices the array up to the length of the bin_time_arr
            data_df[channel] = bin_data_dict[channel][:arr_len]
            
        # Add the 'seed_LD_mode' column from the Bin data
        data_df['seed_LD_mode'] = bin_data_dict['seed_LD_mode'][:arr_len]
        
        # Vectorized Lookup for Task and MFC setpoint
        data_df['Task'] = HK_data['Task'][HK_ind_absolute]
        data_df[f'Cal_{molecule}_MFC_set'] = (
            HK_data[f'Cal_{molecule}_MFC_set'][HK_ind_absolute]
        )
        
        is_cal = data_df['Task'] == cal_task
        cal_transitions = is_cal.astype(int).diff().fillna(0)
        start_indices = cal_transitions[cal_transitions == 1.0].index
        
        file_shift = 0
        
        while len(start_indices) == 0:
            print('\tcannot check for misalignment - '
                  'no calibration found in this file')
            
            file_shift += 1
            
            first_file = soft_restarts['bin_filename'].iloc[i]
            first_file_mask = processing_variables[
                'bin_filename'
            ] == first_file
            
            first_file_restart_index = processing_variables[
                'restart_index'
            ][first_file_mask].iloc[0]
            
            next_restart_index = processing_variables[
                'restart_index'
            ].shift(-file_shift)[first_file_mask].iloc[0]
            
            if first_file_restart_index != next_restart_index:
                print('\tReached end of restart sequence. '
                      'Skipping remaining check.')
                unchecked_restarts.append(
                    (first_file, first_file_restart_index)
                )
                break
            else:
                next_file = processing_variables[
                    'bin_filename'
                ].shift(-file_shift)[first_file_mask].iloc[0]
                next_file_log = processing_variables[
                    'log_start_datetime'
                ].shift(-file_shift)[first_file_mask].iloc[0]
                next_file_date = int(processing_variables[
                    'date'
                ].shift(-file_shift)[first_file_mask].iloc[0])
                
                print(f'\tchecking next file: {next_file}')
                
                log_start_datetime_seconds = format_log_start_datetime(
                    next_file_log
                )
                bin_data = import_bin_data(
                    data_dir, str(next_file_date), next_file
                )
                bin_data_dict = deinterleave_bin_data(
                    bin_data, channel_format, channel_count
                )
                bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(
                    bin_data_dict, log_start_datetime_seconds, HK_data
                )
                HK_time = HK_data['Time_s']
                HK_time_slice = HK_time[HK_start_ind : HK_end_ind + 1]
                HK_ind_relative = np.searchsorted(
                    HK_time_slice, bin_time_arr
                )
                HK_ind_relative = np.clip(
                    HK_ind_relative, 0, len(HK_time_slice) - 1
                )
                HK_ind_absolute = HK_start_ind + HK_ind_relative
                
                data_df = pd.DataFrame()
                arr_len = len(bin_time_arr)
                for channel in channels_to_use:
                    # Slices the array up to the length of the bin_time_arr
                    data_df[channel] = bin_data_dict[channel][:arr_len]
                data_df['seed_LD_mode'] = (
                    bin_data_dict['seed_LD_mode'][:arr_len]
                )
                data_df['Task'] = HK_data['Task'][HK_ind_absolute]
                data_df[f'Cal_{molecule}_MFC_set'] = (
                    HK_data[f'Cal_{molecule}_MFC_set'][HK_ind_absolute]
                )
                is_cal = data_df['Task'] == cal_task
                cal_transitions = is_cal.astype(int).diff().fillna(0)
                start_indices = (
                    cal_transitions[cal_transitions == 1.0].index
                )
                
                
        if len(start_indices) == 0:
            continue
            
        first_start_index = start_indices[0]
        end_indices = cal_transitions[
            (cal_transitions == -1.0) & 
            (cal_transitions.index > first_start_index)
        ].index
        
        if len(end_indices) > 0:
            first_end_index = end_indices[0]
            cal_section = data_df.loc[first_start_index : first_end_index - 1]
        else:
            print('\tmisalignment analysis may be unreliable - '
                  'Calibration runs to the end of the file.')
            cal_section = data_df.loc[first_start_index:]
            
        highest_cal_point = cal_section[
            f'Cal_{molecule}_MFC_set'
        ].max()
        test_section = cal_section[
            cal_section[f'Cal_{molecule}_MFC_set'] == highest_cal_point
        ]
        
        test_section_length = len(test_section)
        start_index = int(test_section_length * 0.2)
        end_index = int(test_section_length * 0.8)
        test_section = test_section[start_index:end_index]
        
        test_section_shifted = test_section.copy()
        
        for channel in channels_to_use:     
            sds_results = []
            
            # For each lag (0, 1, -1)
            for lag in [0, -1, 1, -2, 2]:
                
                sum_of_sds = 0
                test_section_lag = test_section.copy()
                test_section_lag[channel] = (
                    test_section_lag[channel].shift(lag)
                ) 
            
                # For each mode (offline=5, online=6)
                for mode in [5, 6]:

                    values = test_section_lag.loc[
                        test_section_lag['seed_LD_mode'] == mode, channel
                    ]
                    # Calculate sd and sum
                    sum_of_sds += values.std()
                    
                sds_results.append((sum_of_sds, lag))
            
            # Find lag with minimum sum of sds.
            best_lag = min(sds_results)[1]
            
            soft_restarts.loc[i, f'{channel}_shift'] = best_lag    
            
            if best_lag != 0:
                print(f'\tmisaligned data found in {channel}')
        
            test_section_shifted[channel] = (
                test_section[channel].shift(best_lag)
            )
    
        all_lags = soft_restarts.loc[i, [
            f'{channel}_shift' for channel in channels_to_use
        ]]
        if (all_lags == 0).all():
            print('\tNo misaligned data found for any channel.')
    
        if plot:
            
            test_section_plot = test_section.head(20).copy()
            test_section_shifted_plot = test_section_shifted.head(20).copy()
            
            num_plots = len(channels_to_use) + 1
            
            fig, axes = plt.subplots(
                nrows=num_plots, figsize = (10, 3*num_plots)
            )
            fig.suptitle(soft_restarts['bin_filename'][i])
            
            ax = axes[0]
            ax.plot(test_section_shifted_plot.index, 
                    test_section_shifted_plot['seed_LD_mode'])
            ax.set_title('seed_LD_mode')
            
            for i, channel in enumerate(channels_to_use):
                ax = axes[i+1]
                ax.plot(test_section_plot.index, 
                        test_section_plot[channel], 
                        'b-', 
                        label='Original Data')
                ax.plot(test_section_shifted_plot.index, 
                        test_section_shifted_plot[channel], 
                        'r--', 
                        label='Shifted Data')
                ax.set_title(f'{channel}')
                ax.legend()
            plt.tight_layout()
            plt.show()

    # Merge the shift values onto the processing variables df
    shift_columns = [f'{channel}_shift' for channel in channels_to_use]
    columns_to_merge = shift_columns + ['restart_index']

    processing_variables_shifts = pd.merge(
        processing_variables, 
        soft_restarts[columns_to_merge], 
        on='restart_index', 
        how='left'               
    )
    
    # Fill NaN shifts (for files not checked) with 0
    processing_variables_shifts[shift_columns] = (
        processing_variables_shifts[shift_columns].fillna(0)
    )

    # Overwrite the processing variables file with the shifts appended
    processing_variables_shifts.to_csv(
        processing_variables_file_path, index=False
    )
    print('\nProcessing variables file updated to include shift values')
    
    header = (
        '\n\nThe following periods could not be analysed for misaligned data '
        'as no calibration was found between restarts:'
        '\nFirst file after restart,    Restart index'
    )
    data_lines = [
        f"\n{restart[0]},    {restart[1]}" for restart in unchecked_restarts
    ]
    print(header + "".join(data_lines))
