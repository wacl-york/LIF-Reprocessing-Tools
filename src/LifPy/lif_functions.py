import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import islice
from sklearn.linear_model import LinearRegression

from datetime import datetime as dt


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
    
    campaign_subdirs = os.listdir(data_dir)
    day_folders = [folder for folder in campaign_subdirs 
                    if re.match("20[0-9]{2}[0-1][0-9][0-3][0-9]", folder) is not None]
    
    return day_folders

def gen_processing_var(data_dir, day_folders, channel_format, channel_count):
    
    print('Generating Processing Variables')
    
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
            bin_data = import_bin_data(data_dir, bin_df['date'][i]
                                       , bin_df['bin_filename'][i]
                                          )
            bin_data_dict = deinterleave_bin_data(bin_data, channel_format
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
    print(f'\nProcessing variables file created at:\n{processing_variables_file_path}'
          f'\n\n{num_soft_restarts} soft restarts found'
          )
    if num_soft_restarts > 0:
        soft_restarts_file_path = os.path.join(data_dir, 'soft_restarts.txt')
        soft_restart_df.to_csv(soft_restarts_file_path, index=False)
        print(f'\nSoft restarts file created at: \n{soft_restarts_file_path}')
    
def import_HK_data(data_dir, day_folders):
    """
    Imports and concatenates data from multiple Housekeeping (HK) files in a 
    specified directory.

    The function reads all files in the directory, skips a specified number 
    of files at the beginning and end of the list, and aggregates all columns 
    into a single dictionary where keys are the column headers. Files are 
    assumed to be space-delimited (.csv or .txt) with a header row.

    Parameters
    ----------
    HK_file_path : str
        The path to the directory containing the HK data files.
    skip_start : int, optional
        The number of files to skip from the beginning of the file list. 
        Default is 0.
    skip_end : int, optional
        The number of files to skip from the end of the file list. 
        Default is 0.

    Returns
    -------
    dict
        A dictionary where keys are the column headers from the input files, 
        and values are 1D NumPy arrays containing the concatenated data 
        from all selected files for that column.

    Notes
    -----
    The function prints the name of each file being processed.
    It uses a try-except block to initialize the NumPy array for a new header 
    the first time it is encountered.
    """
    
    print('\nreading HK files:\n')
    
    HK_data = {}
    
    for day in day_folders:
        
        file_list = []
        HK_dir = os.path.join(data_dir, day, f"LIFHK_{day}")
        if not os.path.isdir(HK_dir):
            print(f"Warning: Directory not found for day {day}: {HK_dir}")
            continue
        
        file_list.extend(file_name for file_name in os.listdir(HK_dir) if file_name.startswith('LIFHK'))

        if not file_list:
            continue

        for file in file_list:

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
    skip_start : int
        The number of files to skip (discard) from the beginning of the list.
    skip_end : int
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
    
    bin_data = np.fromfile(os.path.join(data_dir, date, f'LIFCnts_{date}', file), dtype='>i2') 
    
    return bin_data

def deinterleave_bin_data(bin_data, channel_format, channel_count,
                           rep_rate_Hz=200000):
    """
    De-interleaves raw binary data from multiple channels, combines high/low 
    bytes where necessary, applies calibration, and calculates derived metrics.

    Raw data is assumed to be interleaved by channel. The function processes 
    each channel based on the provided format dictionary and calculates 
    corrected and normalized counts.

    Parameters
    ----------
    binary_data : numpy.ndarray
        A 1D array of raw, interleaved data points.
    channel_format : dict
        A dictionary mapping output channel names (str) to their input indices 
        (int) or high/low index pairs (list of int).
        - Value as int (e.g., 2): Single channel index.
        - Value as list [int, int] (e.g., [3, 4]): High-byte, Low-byte indices.
    channel_count : int
        The total number of channels (interleaving factor).
    rep_rate_Hz : int, optional
        The repetition rate (in Hz) of the laser, used to 
        calculate the maximum theoretical count rate. Default is 200000.

    Returns
    -------
    tuple
        - binary_data_dict (dict): A dictionary containing all de-interleaved, 
          combined, and derived data channels. Keys include original channel 
          names, plus '_lin' (linearized) and '_norm' (normalized) for count 
          data.
        - lag (float or np.nan): The time difference (in ms) of a detected 
          large time step, or NaN if no significant lag is found.

    Notes
    -----
    - **De-interleaving:** 
        Uses slicing (frames[idx::channel_count]) to separate the interleaved 
        channels.
    - **Byte Combination (if channel_ID is a list):** 
        Combines 16-bit high-byte (multiplied by 65536) and 16-bit low-byte 
        channels to form a 32-bit value.
    - **Laser Power:** 
        'laser_pwr_PT0' is converted from system units to mW by dividing by 
        100,000.
    - **Count Correction:**
        - Values exceeding the maximum theoretical count rate 
          (rep_rate_Hz / 100) are set to NaN.
        - Linearized counts (`_lin`) are calculated using:
          -ln(1 - (cts / max_cts)) * max_cts
        - Normalized counts (`_norm`) are calculated using:
          cts / laser power 
    - **Lag Calculation:** 
        Attempts to find a time step close to 20,000 ms (20 seconds) in the 
        'time_ms' channel to identify a potential data break or lag.
    """
    
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
    
    return bin_data_dict
 
def gen_output_file(data_dir, data_freq, file, channel_format, HK_headers_dict
                    , HK_data, date, bin_data_dict):
   """
    Creates and initializes the output text file, writes metadata and all 
    headers, and generates a NaN placeholder string for non-data periods.

    The function determines the next available file index, opens the new file, 
    writes standard metadata, generates the full combined header string 
    (General, Signal Counts, and validated Housekeeping headers), and then 
    writes the headers to the file.

    Parameters
    ----------
    working_dir : str
        The root directory of the project, used to locate the output folder 
        and the 'lif_metadata.txt' template file.
    file : str
        The name of the current binary input file (used for naming the output 
        file).
    channel_format : dict
        A dictionary defining the data channels, used to generate headers for 
        signal counts, linearized, and normalized channels.
    HK_headers_dict : dict
        A dictionary of desired Housekeeping headers. This dictionary is 
        checked against HK_data and modified in-place to remove non-existent 
        headers.
    HK_data : dict
        The dictionary containing all Housekeeping data (used to validate that 
        requested headers in HK_headers_dict actually exist).

    Returns
    -------
    str
        A comma-separated string (nan_data) of placeholder values (-9999) 
        matching the width of the data columns defined by all_headers, used 
        to fill periods of missing or invalid data.

    """
   
   output_dir = os.path.join(data_dir, date, f'LIFProcessed_{date}')
   try:
     os.makedirs(output_dir, exist_ok=True)
   except OSError as e:
     print(f"Error creating directory {output_dir}: {e}")
   
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
                           
       cts_labels = [name.split('_')[0] + ('_' + name.split('_')[1] if len(name.split('_')) > 2 else '')
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
 
    bin_time_arr = \
        (bin_data_dict['time_ms'] / 1000) + log_start_datetime_seconds
        
    HK_start_ind = find_min_ind(bin_time_arr[0], HK_data['Time_s']) 
    
    HK_end_ind = find_min_ind(bin_time_arr[-1], HK_data['Time_s'])
    
    if HK_start_ind == HK_end_ind:
        print('\nThe log_start_datetime is incorrect')
        print(bin_time_arr[0])
        sys.exit(1)
        
    return bin_time_arr, HK_start_ind, HK_end_ind 
    
def gen_output_data(file, channel_format, data_dir, bin_data_dict, HK_data, HK_headers_dict, 
                    data_freq, bin_time_arr, HK_start_ind, HK_end_ind, 
                    nan_data, processed_file
                    , histograms):
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
    processing_variables = pd.read_csv(os.path.join(
        data_dir, 'processing_variables.txt')
        )
    this_file_index = processing_variables[
        processing_variables['bin_filename'] == file].index[0]
    
    for channel in channel_format:
        shift_column = f'{channel}_shift'
        if shift_column in processing_variables.columns:
            shift_value = int(processing_variables[shift_column][this_file_index])
        else:
            shift_value = 0
        data_array = bin_data_dict[channel]
        original_length = len(data_array)
        if shift_value == 0:
            continue
        elif shift_value > 0:
        # Positive shift (data moves right): Pad the beginning (left) with NaNs.
            pad_width = (shift_value, 0)
            shifted_array = np.pad(
                data_array, 
                pad_width, 
                mode='constant', 
                constant_values=np.nan
            )
            # Slice off the end to maintain original length
            bin_data_dict[channel] = shifted_array[:original_length]
        elif shift_value < 0:
        # Negative shift (data moves left): Pad the end (right) with NaNs.
            abs_shift = abs(shift_value)
            pad_width = (0, abs_shift)
            shifted_array = np.pad(
                data_array, 
                pad_width, 
                mode='constant', 
                constant_values=np.nan
            )
            # Slice off the beginning to maintain original length
            bin_data_dict[channel] = shifted_array[abs_shift:]   
    
    
    
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
         
# =============================================================================
#             gen_write = f"{curr_time},{laser_pwr_PT0[i]},{seed_LD_mode[i]},{seed_LD_current}"
#             cts_values_at_i = [bin_data_dict[name][i] \
#                                for name in bin_keys if '_counts' in name \
#                                    and 'ref' not in name]
# =============================================================================
            bin_values_at_i = [bin_data_dict[name][i] \
                               for name in bin_keys]
            bin_write = ','.join(map(str, bin_values_at_i))
            #cts_write = ','.join(map(str, cts_values_at_i))
            HK_write = ','.join(
                [f'{HK_data[key][HK_ind]}' for key in HK_headers]
            )
       
            output_lines.append(f'{bin_write},{HK_write}\n')   
         
    processed_file.write(''.join(output_lines))     

    if processed_points > 0:
        
        print(f'\npercentage of no dropped online points = {((no_dropped_online/processed_points)*100): .2f}'
              f'\npercentage of one dropped online point = {((one_dropped_online/processed_points)*100): .2f}'
              f'\npercentage of two dropped online points = {((two_dropped_online/processed_points)*100): .2f}'
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
                          , channel_format, data_dir, HK_data, data_freq=10
                          , channel_count=10, histograms=False):
    """
    This function calls all of the other sub-functions above, to read and 
    process the binary and HK files into counts data in a readable csv format.
    
    Parameters
    ----------
    log_start_datetime : str
        The log time of the corresponding LIF system restart.
    date : str
        The date associated with the data being processed in the form 'YYYYMMDD'
    HK_headers_dict : dict
        A dictionary containing information about the Housekeeping data 
        headers/channels.
    channel_format : list
        A list or array defining the format and types of the data channels 
        within the binary files.
    working_dir : str
        The base directory where the binary data and output files are located.
    bin_file_path : str, optional
        The sub-directory path within `working_dir` where the raw binary files 
        reside. The default is 'data_bin'.
    HK_file_path : str, optional
       The sub-directory path within `working_dir` where the HK files reside. 
       The default is 'data_HK'. 
    data_freq : int, optional
        The frequency of data to be returned, can be 10 or 100 HZ. The default 
        is 10.
    skip_start_HK : int, optional
        Number of HK data files to skip at the beginning of the specified 
        folder. The default is 0.
    skip_end_HK : int, optional
        Number of HK data files to skip at the end of the specified folder. The 
        default is 0.
    skip_start_bin : int, optional
        Number of binary data files to skip at the beginning of the specified 
        folder. The default is 0.
    skip_end_bin : int, optional
        Number of binary data files to skip at the end of the specified folder. The 
        default is 0
    lag_override : int, optional
        A manual override value for the time lag correction, if needed. The 
        default is 0.
    channel_count : int, optional
        The total number of channels in the binary data. The default is 10.
    histograms : bool, optional
        Flag to indicate whether histograms should be generated during 
        processing. The default is False.
    
    Returns
    -------
    None
        The function performs file operations and generates output files but does not
        explicitly return a value.
    
    """
    
    print('\nReprocessing file %s' % file)
    
    log_start_datetime_seconds = format_log_start_datetime(
        log_start_datetime
        )
    bin_data = import_bin_data(
        data_dir, date, file
        )
    bin_data_dict = deinterleave_bin_data(
        bin_data, channel_format, channel_count
        )
    processed_file, nan_data = gen_output_file(
        data_dir, data_freq, file, channel_format, HK_headers_dict
        , HK_data, date, bin_data_dict
        )
    bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(
        bin_data_dict, log_start_datetime_seconds, HK_data
        )
    gen_output_data(
        file, channel_format, data_dir, bin_data_dict, HK_data, HK_headers_dict, data_freq
        , bin_time_arr, HK_start_ind, HK_end_ind, nan_data, processed_file
        , histograms
        )

def misaligned_counts(data_dir, day_folders, channel_format, channel_count
                      , molecule, cal_task=2, plot=False):

    excluded_channels = ['laser_pwr_PT0', 'seed_LD_mode', 'time_ms']
    channels_to_use = [key for key in channel_format.keys() if key not in excluded_channels]
    
    processing_variables_file_path = os.path.join(data_dir, 'processing_variables.txt')
    processing_variables = pd.read_csv(processing_variables_file_path).copy()
    soft_restarts_file_path = os.path.join(data_dir, 'soft_restarts.txt')
    if not os.path.exists(soft_restarts_file_path):
        print('\nNo soft restarts identified - misalignment check not required')
        return
    soft_restarts = pd.read_csv(os.path.join(data_dir, 'soft_restarts.txt')).copy()
    HK_data = import_HK_data(data_dir, day_folders)
    
    for channel in channels_to_use:
        soft_restarts[f'{channel}_shift'] = 0
        
    unchecked_restarts = []

    # Loop through each soft restart 
    for i in soft_restarts.index:
        
        print(f"\nchecking for misaligned data in file: {soft_restarts['bin_filename'][i]}")
        
        log_start_datetime_seconds = format_log_start_datetime(soft_restarts['log_start_datetime'][i])
        bin_data = import_bin_data(data_dir, str(soft_restarts['date'][i]), str(soft_restarts['bin_filename'][i]))
        bin_data_dict = deinterleave_bin_data(bin_data, channel_format, channel_count)
        bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data)
        
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
        data_df[f'Cal_{molecule}_MFC_set'] = HK_data[f'Cal_{molecule}_MFC_set'][HK_ind_absolute]
        
        is_cal = data_df['Task'] == cal_task
        cal_transitions = is_cal.astype(int).diff().fillna(0)
        start_indices = cal_transitions[cal_transitions == 1.0].index
        
        file_shift = 0
        
        while len(start_indices) == 0:
            print('\tcannot check for misalignment - no calibration found in this file')
            
            file_shift += 1
            
            first_file = soft_restarts['bin_filename'].iloc[i]
            first_file_mask = processing_variables['bin_filename'] == first_file
            
            first_file_restart_index = processing_variables['restart_index'][first_file_mask].iloc[0]
            next_file_restart_index = processing_variables['restart_index'].shift(-file_shift)[first_file_mask].iloc[0]
            if first_file_restart_index != next_file_restart_index:
                print('\tReached end of restart sequence. Skipping remaining check.')
                unchecked_restarts.append((first_file, first_file_restart_index))
                break
            else:
                next_file = processing_variables['bin_filename'].shift(-file_shift)[first_file_mask].iloc[0]
                next_file_log = processing_variables['log_start_datetime'].shift(-file_shift)[first_file_mask].iloc[0]
                next_file_date = int(processing_variables['date'].shift(-file_shift)[first_file_mask].iloc[0])
                
                print(f'\tchecking next file: {next_file}')
                
                log_start_datetime_seconds = format_log_start_datetime(next_file_log)
                bin_data = import_bin_data(data_dir, str(next_file_date), next_file)
                bin_data_dict = deinterleave_bin_data(bin_data, channel_format, channel_count)
                bin_time_arr, HK_start_ind, HK_end_ind = align_bin_HK(bin_data_dict, log_start_datetime_seconds, HK_data)
                HK_time = HK_data['Time_s']
                HK_time_slice = HK_time[HK_start_ind : HK_end_ind + 1]
                HK_ind_relative = np.searchsorted(HK_time_slice, bin_time_arr)
                HK_ind_relative = np.clip(HK_ind_relative, 0, len(HK_time_slice) - 1)
                HK_ind_absolute = HK_start_ind + HK_ind_relative
                
                data_df = pd.DataFrame()
                for channel in channels_to_use:
                     # Slices the array up to the length of the bin_time_arr
                     data_df[channel] = bin_data_dict[channel][:len(bin_time_arr)]
                data_df['seed_LD_mode'] = bin_data_dict['seed_LD_mode'][:len(bin_time_arr)]
                data_df['Task'] = HK_data['Task'][HK_ind_absolute]
                data_df[f'Cal_{molecule}_MFC_set'] = HK_data[f'Cal_{molecule}_MFC_set'][HK_ind_absolute]
                is_cal = data_df['Task'] == cal_task
                cal_transitions = is_cal.astype(int).diff().fillna(0)
                start_indices = cal_transitions[cal_transitions == 1.0].index
                
                
        if len(start_indices) == 0:
            continue
        
        first_start_index = start_indices[0]
        end_indices = cal_transitions[(cal_transitions == -1.0) & (cal_transitions.index > first_start_index)].index
        if len(end_indices) > 0:
            first_end_index = end_indices[0]
            cal_section = data_df.loc[first_start_index : first_end_index - 1]
        else:
            print('\tmisalignment analysis may be unreliable - Calibration runs to the end of the file.')
            cal_section = data_df.loc[first_start_index:]
            
        highest_cal_point = cal_section[f'Cal_{molecule}_MFC_set'].max()
        test_section = cal_section[cal_section[f'Cal_{molecule}_MFC_set'] == highest_cal_point]
        
        test_section_length = len(test_section)
        start_index = int(test_section_length * 0.2)
        end_index = int(test_section_length * 0.8)
        test_section = test_section[start_index:end_index]
        
        test_section_shifted = test_section.copy()
        
        for channel in channels_to_use:     
            #sum_of_sds = 0
            sds_results = []
           
            # For each lag (0, 1, -1)
            #these are the defined shifts. These may change.
            for lag in [0, -1, 1]:
                
                sum_of_sds = 0
                test_section_lag = test_section.copy()
                test_section_lag[channel] = test_section_lag[channel].shift(lag) 
           
                # For each mode
                #offline ==5
                #online ==6
                for mode in [5, 6]:

                    values = test_section_lag.loc[test_section_lag['seed_LD_mode'] == mode, channel]
                    # Calculate sd
                    #add it on to the 0 from the sum_of_stds and then also the new calulated values
                    sum_of_sds += values.std()
                
                sds_results.append((sum_of_sds, lag))
            #print(channel, '', sds_results)

               
            # Find lag with minimum sum of sds. NB: min() of a list of tuples sorts by
            # first tuple entry, i.e. sd.
            best_lag = min(sds_results)[1]
            
            soft_restarts.loc[i, f'{channel}_shift'] = best_lag    
            
            if best_lag != 0:
                print(f'\tmisaligned data found in {channel}')
        
            test_section_shifted[channel] = test_section[channel].shift(best_lag)
    
        all_lags = soft_restarts.loc[i, [f'{channel}_shift' for channel in channels_to_use]]
        if (all_lags == 0).all():
            print('\tNo misaligned data found for any channel.')
    
        if plot:
            
            test_section_plot = test_section.head(20).copy()
            test_section_shifted_plot = test_section_shifted.head(20).copy()
            
            num_plots = len(channels_to_use) + 1
            
            fig, axes = plt.subplots(nrows=num_plots, figsize = (10, 3*num_plots))
            fig.suptitle(soft_restarts['bin_filename'][i])
            ax = axes[0]
            ax.plot(test_section_shifted_plot.index, test_section_shifted_plot['seed_LD_mode'])
            ax.set_title('seed_LD_mode')
            for i, channel in enumerate(channels_to_use):
                ax = axes[i+1]
                ax.plot(test_section_plot.index, test_section_plot[channel], 'b-', label='Original Data')
                ax.plot(test_section_shifted_plot.index, test_section_shifted_plot[channel], 'r--', label='Shifted Data')
                ax.set_title(f'{channel}')
                ax.legend()
            plt.tight_layout()
            plt.show()

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
    #processing_variables_shifts = processing_variables_shifts.drop(columns=['restart_index'])

    # Overwrite the processing variables file with the shifts appended
    processing_variables_shifts.to_csv(processing_variables_file_path, index=False)
    print('\nProcessing variables file updated to include shift values')
    header = ('\n\nThe following periods could not be analysed for misaligned data as no calibration was found between restarts:'
              '\nFirst file after restart,    Restart index')
    data_lines = [f"\n{restart[0]},    {restart[1]}" for restart in unchecked_restarts]
    print(header + "".join(data_lines))


"""
This section contains all of the sub-functions that are used in the 
interpretation of counts data to give mixing ratios.

"""

def read_processed_files(data_dir, day_folders):
    
    print('\nreading Processed files:\n')
    
    dfs = []
    
    for day in day_folders:
        
        file_list = []
        processed_dir = os.path.join(data_dir, day, f"LIFProcessed_{day}")
        if not os.path.isdir(processed_dir):
            print(f"Warning: Directory not found for day {day}: {processed_dir}")
            continue
        
        file_list.extend(file_name for file_name in os.listdir(processed_dir) 
                         if file_name.startswith('20') 
                         and file_name.endswith('.txt'))

        if not file_list:
            continue
        
        for file in file_list:
            
            print(f'\r{file}', end='')
            
            file_path = os.path.join(processed_dir, file)
            try:
                df = pd.read_csv(file_path, header=7)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}") 
                
    if not dfs:
        print("No files found or processed.")
        return pd.DataFrame()      
    
    dfs_filtered = [df for df in dfs if not df.empty]
    
    cts_data = pd.concat(dfs_filtered, ignore_index=True)

    #replacing -9999 with nan
    cts_data.replace(-9999, np.nan, inplace=True)
    cts_data.dropna(inplace=True)
    cts_data.reset_index(drop=True, inplace=True)
    # converting date_time
    cts_data.rename(columns={'mac_time_s': 'Date_time'}, inplace=True)
    cts_data['Date_time'] = (pd.to_datetime(cts_data['Date_time'], unit='s') 
                            - pd.to_timedelta(2082844800, unit='s'))
    cts_data['Date_time'] = pd.to_datetime(cts_data['Date_time'])
    cts_data = cts_data.sort_values(by='Date_time')

    return cts_data

def ref_normalise(data, channels):
    
    data = data.copy()
    
    for channel in channels:
        
        data[f'{channel}_diff_cts_ref_norm'] = data[f'{channel}_diff_cts_norm'] / data['ref_diff_cts_norm']
    
    return data 

def set_flags(data, pre_TS, post_TS, pre_PF, post_PF, ref_cts_limit):
    
    data = data.reset_index(drop=True)
    peak_flag_array = np.zeros(len(data), dtype=int)
    data["Task_Change"] = data.Task.shift() != data.Task
    task_switch_indices = data.index[data['Task_Change']].to_numpy()
    
    condition_mask = data['ref_diff_cts_norm'] < ref_cts_limit
    condition_indices = data.index[condition_mask].to_numpy()
    
    for index in condition_indices:
        start = max(0, index - pre_PF)
        end = min(len(data) - 1, index + post_PF)
        
        peak_flag_array[start:end+1] = 1
    
    data['Peak_find_flag'] = peak_flag_array
    
    task_array = data['Task'].to_numpy()
    
    for index in task_switch_indices:
        if index > 0:
            start = max(0, index - pre_TS)
            end = min(len(data) - 1, index + post_TS)

            task_array[start:end+1] = 8 
            
    data['Task'] = task_array
    
    data.index = data['Date_time']
    data = data.drop(columns=['Task_Change'])
    
    return data

def zero_correct_average(data, channels, plot=False):
    
    # use a mask to select all of the zero data associated with task 4
    # set the index to Date_time for averaging later
    cts_data_zero = data.copy()
    start_of_zero = (cts_data_zero['Task'] == 4) & (cts_data_zero['Task'].shift(1) != 4)
    cts_data_zero['zero_number'] = start_of_zero.cumsum()
    cts_data_zero = cts_data_zero[(cts_data_zero['Task']==4) & (cts_data_zero['Peak_find_flag']==0)]
    grouped_zeros = cts_data_zero.groupby('zero_number')
    
    if 'sig_B' in channels:
        
        cts_data_zero['sig_B_diff_cts_ref_norm'] = np.where(
            (cts_data_zero['BLC_0_flag'] == 1.0) & (cts_data_zero['BLC_1_flag'] == 1.0)
            , cts_data_zero['sig_B_diff_cts_ref_norm']
            , np.nan
            )
    
    for channel in channels:

        column_name = f'{channel}_diff_cts_ref_norm'
        
        zero_stats = grouped_zeros.agg({
            column_name: ['mean', 'std', 'count'],
            'Date_time': 'mean'
            }).dropna()
        
        # Flattening the MultiIndex columns
        zero_stats.columns = ['_'.join(col).strip() for col in zero_stats.columns.values]
        
        # Renaming for clarity
        zero_stats = zero_stats.rename(columns={
            'Date_time_mean': 'zero_midpoint'
        })
        
        mean_zero = zero_stats[f'{column_name}_mean'].mean()
        std_zero = zero_stats[f'{column_name}_mean'].std()
        lower_limit = mean_zero - 3*std_zero
        upper_limit = mean_zero + 3*std_zero
        spike_mask = ((zero_stats[f'{column_name}_mean'] > lower_limit)
            & (zero_stats[f'{column_name}_mean'] < upper_limit))
        
        zero_stats = zero_stats[spike_mask]
        
        mean_zero_spikes_removed = zero_stats[f'{column_name}_mean'].mean()
        print(f'\n{channel}:\nmean zero before spike removal = {mean_zero}'
              f'\nmean zero after spike removal = {mean_zero_spikes_removed}')
        
        mean_correction = mean_zero_spikes_removed # The overall mean after spike removal
        correction_values = np.full(len(data), mean_correction)
        
        data[f'{channel}_zero_offset'] = correction_values
        data[f'{channel}_diff_cts_ref_norm_zero_corr'] = data[column_name] - correction_values
        print(f'zero correction applied to {channel}')
        
        if plot:
            
            fig, ax = plt.subplots( 2, 1, figsize=(12, 8))
            
            ax[0].errorbar(zero_stats['zero_midpoint']
                             , zero_stats[f'{column_name}_mean']
                             , yerr=zero_stats[f'{column_name}_std']
                             , linestyle=''
                             , marker='o'
                             , markersize=2
                             , capsize=2
                             , label='zero measurement means, +/- 1std'
                             )
            #ax.plot(cts_data_zero['Date_time'], cts_data_zero[column_name], label='raw zero data')
            ax[0].plot(data['Date_time'], data[f'{channel}_zero_offset'], label='zero correction')
            ax[0].set_xlabel('Date_time')
            ax[0].set_ylabel(column_name)
            ax[0].set_title(f'{channel} zero correction')
            ax[0].legend()
            
            ax[1].hist(zero_stats[f'{column_name}_mean'], bins=50)
            ax[1].set_xlabel(column_name)
            
            plt.show()

    return data
    
def analyse_cals(data, plot, max_conc, channels, data_dir, molecule):
    """
    Analyzes calibration data for a specified cell by identifying
    individual calibration events, applying data cleaning, and performing
    linear regression on both standard and ref-normalised signals.
    
    The function identifies calibration groups based on flow changes and
    filters out transient signal/flow switching periods before fitting.
    Regression results (R-squared, slope, intercept) are summarized, printed,
    saved to a file, and optionally plotted.
    
    Parameters
    ----------
    all_data : pandas.DataFrame
        The complete dataset.
    plot : bool
        If True, generates and displays a 4xN subplot figure (where N is the
        number of calibration events) showing signal time series and
        regression fits.
    max_conc : float
        The maximum true concentration (in ppt) to include in the linear
        regression analysis. Points above this value are excluded.
    cell : str
        The identifier for the measurement cell being analyzed (e.g., 'A', 'B').
        Used to dynamically select column names and output filenames.
    
    Returns
    -------
    Std_cal_summary : pandas.DataFrame
        A summary table of linear regression results for the standard
        (non-ref-normalised) signal
    Refnorm_cal_summary : pandas.DataFrame
        A summary table of linear regression results for the ref-normalised
        signal.
    
    Notes
    -----
    - Calibration groups are defined by a gap of non-calibration points 
      (3000 points) followed by a calibration point (Task == 5).
    - Data points are filtered to exclude flow switching transients and signal
      instabilities based on rolling statistics and flow rate changes 
      (diff > 0.05).
    - Regression is performed on data points starting at index 300 of the 
      filtered calibration period to ensure steady-state conditions.
    """
    for channel in channels:
    
        file_path = os.path.join(data_dir, f'{channel}_cal_data.txt')
        
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as txtfile:
                    fieldnames = ['cal_start_date_time', 'avg_lsr_pwr', 'R2'
                                  , 'slope', 'intercept', 'R2_ref_norm'
                                  , 'slope_ref_norm', 'intercept_ref_norm']
                    header_row = ','.join(fieldnames)
                    txtfile.write(header_row + '\n')    
        
        print(f'\nidentifying cals, {channel}')
        
        cts_diff_v = f'{channel}_diff_cts'
        cts_diff_refnorm_v = f'{channel}_diff_cts_ref_norm'
        data = data[[cts_diff_v, cts_diff_refnorm_v, f'{molecule}_mr', 'Task'
                         , f'Cal_{molecule}_MFC_Read', 'Cal_SB_MFC_Read', f'Cal_{molecule}_MFC_set'
                         , 'Date_time', 'lsr_pwr_mW']].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.reset_index(drop=True)
        data['cal_sig_diff_cts_ref_norm'] = data[cts_diff_refnorm_v].where(
            data['Task'] == 5
            )
        data['cal_true_ppt'] = data['NO_mr'].where(
            (data.Task == 5) & (data['Cal_SB_MFC_Read'] < 0.01)
            )
        data['cal_group'] = np.nan
        data['cal_start_time'] = pd.NaT
        cal_num = 0
        cal_group_start_times = {}
        tot_steps = len(data.index) - 1
    
        
        for i in data.index:
            
            if i % 1000 == 0:
                print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100)
                      , end='')
            
            if (i > 0 and pd.notnull(data['cal_true_ppt'][i]) 
                and pd.isnull(data['cal_true_ppt'][max(0, i-3000):i].mean())):
                cal_num += 1
                cal_group_start_times[cal_num] = data.loc[i, 'Date_time']
                data.loc[i, 'cal_start_time'] = data.loc[i, 'Date_time']
            if (pd.notnull(data['cal_true_ppt'][i]) 
                and pd.notnull(data['cal_true_ppt'][max(0, i-3001):max(0, i-100)].mean())):
                data.loc[i, 'cal_group'] = cal_num
         
            
    
        backward_mean = data['cal_true_ppt'].rolling(window=200).mean().shift(1)
        backward_std = data['cal_true_ppt'].rolling(window=200).std().shift(1)
        forward_mean = data['cal_true_ppt'][::-1].rolling(window=200).mean()[::-1]
        mask = (forward_mean > backward_mean + backward_std/2) | \
           (forward_mean < backward_mean - backward_std/2)
        data.loc[mask, 'cal_true_ppt'] = np.nan
    
        print('\nnumber of cals =', cal_num )
    
        Refnorm_cal_vars = {}
        std_cal_vars = {}
        fig, axs = None, None
    
        if plot:
            fig, axs = plt.subplots(4, cal_num, figsize=(6 * cal_num, 12))
    
        for cal in range(1,cal_num+1):
            
            print(f'\ranalysing cal {cal}', end='')
            
            current_cal_start_time = cal_group_start_times.get(cal, None)
            
            cal_tmp_df = data[(data['cal_group'] == cal)].copy().dropna(
               subset=[cts_diff_v, cts_diff_refnorm_v, 'NO_mr', 'cal_true_ppt'
                       , 'Cal_NO_MFC_set', 'lsr_pwr_mW']
            )
            
            avg_lsr_pwr = cal_tmp_df['lsr_pwr_mW'].mean()
            
            cal_tmp_df['point_filter'] = 0
            cal_tmp_df['cal_flow_diff'] = cal_tmp_df['Cal_NO_MFC_set'].diff().abs()
            for cal_pt in cal_tmp_df.index:
                if (cal_tmp_df['cal_flow_diff'][cal_pt] > 0.05):
                    cal_tmp_df.loc[cal_pt-1:cal_pt+10,'point_filter'] = 1
            cal_tmp_df = cal_tmp_df[(cal_tmp_df['point_filter'] == 0) 
                                    & (cal_tmp_df['cal_true_ppt'] < max_conc)]
            if cal_tmp_df.shape[0] > 100:
                X = cal_tmp_df['cal_true_ppt'][300:].values.reshape(-1, 1)
                Y = cal_tmp_df['cal_sig_diff_cts_ref_norm'][300:].values.reshape(-1, 1)
                linear_regressor = LinearRegression()
                reg = linear_regressor.fit(X, Y)
                Y_pred = linear_regressor.predict(X)
                Norm_cal_dict = {}
                Norm_cal_dict['cal_start_date_time'] = current_cal_start_time
                Norm_cal_dict['avg_lsr_pwr'] = avg_lsr_pwr
                Norm_cal_dict['R2'] = reg.score(X,Y)
                Norm_cal_dict['Slope'] = reg.coef_[0,0]
                Norm_cal_dict['Intercept'] = reg.intercept_[0]
                Refnorm_cal_vars[cal] =  Norm_cal_dict
                
                Y2 = cal_tmp_df[cts_diff_v][300:].values.reshape(-1, 1)
                linear_regressor = LinearRegression()
                reg2 = linear_regressor.fit(X, Y2)
                Y2_pred = linear_regressor.predict(X)
                cal_dict = {}
                cal_dict['cal_start_date_time'] = current_cal_start_time
                cal_dict['avg_lsr_pwr'] = avg_lsr_pwr
                cal_dict['R2'] = reg2.score(X,Y2)
                cal_dict['Slope'] = reg2.coef_[0,0]
                cal_dict['Intercept'] = reg2.intercept_[0]
                std_cal_vars[cal] = cal_dict
                
                
                # cal_data = pd.DataFrame({'X':cal_tmp_df['Cal_true_ppt'][30:], 'Y':cal_tmp_df['Cal_Sig_diff_cts_ref_norm'][30:]})
                # cal_data.to_csv('Cell_'+cell+'_Cal_data_'+str(cal)+'.csv')
            if cal in Refnorm_cal_vars and cal in std_cal_vars:
                
                if current_cal_start_time is not None:
                    
                    new_data_to_append = {
                        'cal_start_date_time': 
                            [current_cal_start_time],
                        'avg_lsr_pwr':
                            [avg_lsr_pwr],
                        'R2': 
                            [std_cal_vars[cal]['R2']],
                        'slope': 
                            [std_cal_vars[cal]['Slope']],
                        'intercept': 
                            [std_cal_vars[cal]['Intercept']],
                        'R2_ref_norm': 
                            [Refnorm_cal_vars[cal]['R2']],
                        'slope_ref_norm': 
                            [Refnorm_cal_vars[cal]['Slope']],
                        'intercept_ref_norm': 
                            [Refnorm_cal_vars[cal]['Intercept']]
                    }
                    
                new_data_to_append_df = pd.DataFrame(new_data_to_append)
                
                new_data_to_append_df.to_csv(
                    file_path, mode='a', header=False
                    , index=False, sep=','
                    )
                
                
                if plot:
    
                    # --- First Plot ---
                    # Plot on the first subplot (axs[0])
                    axs[0, cal-1].plot(cal_tmp_df.index, cal_tmp_df[cts_diff_v])
                    axs[0, cal-1].set_title(f'Standard cal {cal} {channel}')
    
                    # --- Second Plot ---
                    # Plot on the second subplot (axs[1])
                    axs[1,cal-1].scatter(X, Y2)
                    axs[1,cal-1].plot(X, Y2_pred, color='red')
                    axs[1,cal-1].set_title(f'Standard cal {cal} {channel}')
        
                    # --- Third Plot ---
                    # Plot on the third subplot (axs[2])
                    axs[2,cal-1].plot(
                        cal_tmp_df.index, cal_tmp_df['cal_sig_diff_cts_ref_norm']
                        )
                    axs[2,cal-1].set_title(f'Ref norm cal {cal} {channel}')
    
                    # --- Fourth Plot ---
                    # Plot on the fourth subplot (axs[3])
                    axs[3,cal-1].scatter(X, Y)
                    axs[3,cal-1].plot(X, Y_pred, color='red')
                    axs[3,cal-1].set_title(f'Ref norm cal {cal} {channel}')
        
        # Display the combined figure
        if plot:
            plt.tight_layout()
            plt.show()
        
        Std_cal_summary = pd.DataFrame(std_cal_vars).transpose()
        print(f'{channel} Non ref normalised cals')
        print(Std_cal_summary)
        if (100*(Std_cal_summary['Slope'].std()/Std_cal_summary['Slope'].mean())) < 5:
            print(f'{channel} Cal slope standard deviation < 5% of mean')
        else:
            print(f'{channel} Cal slope standard deviation greater than 5% of mean')
    
        Refnorm_cal_summary = pd.DataFrame(Refnorm_cal_vars).transpose()
        print(f'{channel} Reference cell normalised cals')
        print(Refnorm_cal_summary)
        if (100*(Refnorm_cal_summary['Slope'].std()/Refnorm_cal_summary['Slope'].mean())) < 5:
            print(f'{channel} Ref norm cal slope standard deviation < 5% of mean')
        else:
            print(f'{channel} Ref norm cal slope standard deviation greater than 5% of mean')    
        
def analyse_BLC_cals(all_data, data_dir, plot): 
    
    
    file_path = os.path.join(data_dir, 'BLC_cal_data.txt')
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as txtfile:
                fieldnames = ['cal_start_date_time', 'conversion_efficiency', 'BLC_V']
                header_row = ','.join(fieldnames)
                txtfile.write(header_row + '\n') 
    
    data = all_data[['sig_B_diff_cts', 'sig_B_diff_cts_ref_norm', 'Task', 'Date_time', 'BLC_0_flag', 'BLC_1_flag']].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.reset_index(drop=True)
    data['BLC_Cal_sig_diff_cts_ref_norm'] = data['sig_B_diff_cts_ref_norm'].where(data['Task'] == 1)
    data['BLC_Cal_group'] = np.nan
    data['BLC_cal_start_time'] = pd.NaT
    BLC_cal_num = 0
    BLC_cal_group_start_times = {}  
    
    for i in data.index:
        if (i > 0 and pd.notnull(data['BLC_Cal_sig_diff_cts_ref_norm'][i]) and pd.isnull(data['BLC_Cal_sig_diff_cts_ref_norm'][max(0, i-3000):i].mean())):
            BLC_cal_num += 1
            BLC_cal_group_start_times[BLC_cal_num] = data.loc[i, 'Date_time']
        if (pd.notnull(data['BLC_Cal_sig_diff_cts_ref_norm'][i]) and pd.notnull(data['BLC_Cal_sig_diff_cts_ref_norm'][max(0, i-3010):max(0, i-100)].mean())):
            data.loc[i, 'BLC_Cal_group'] = BLC_cal_num
            
    fig, axs = plt.subplots(1, BLC_cal_num, figsize=(6*BLC_cal_num, 4))
    
    for cal in range(1,BLC_cal_num+1):
        
        cal_tmp_df = data[(data['BLC_Cal_group'] == cal)].copy()#.dropna()
        cal_tmp_df['cal_section'] = 0
        cal_tmp_df.iloc[300:2660, cal_tmp_df.columns.get_loc('cal_section')] = 1
        cal_tmp_df.iloc[3260:5620, cal_tmp_df.columns.get_loc('cal_section')] = 2
        cal_tmp_df.iloc[6520:8580, cal_tmp_df.columns.get_loc('cal_section')] = 3
        cal_tmp_df.iloc[9280:11540, cal_tmp_df.columns.get_loc('cal_section')] = 4
        gpt_off_blc_off = cal_tmp_df['BLC_Cal_sig_diff_cts_ref_norm'].where(cal_tmp_df['cal_section'] == 1).mean()
        gpt_off_blc_on = cal_tmp_df['BLC_Cal_sig_diff_cts_ref_norm'].where(cal_tmp_df['cal_section'] == 2).mean()
        gpt_on_blc_off = cal_tmp_df['BLC_Cal_sig_diff_cts_ref_norm'].where(cal_tmp_df['cal_section'] == 3).mean()
        gpt_on_blc_on = cal_tmp_df['BLC_Cal_sig_diff_cts_ref_norm'].where(cal_tmp_df['cal_section'] == 4).mean()
        conversion_efficiency = 1- ((gpt_off_blc_on - gpt_on_blc_on)/(gpt_off_blc_off - gpt_on_blc_off))
        print('conversion efficiency cal ' + str(cal) + ':' + str(conversion_efficiency))

        current_cal_start_time = BLC_cal_group_start_times.get(cal, None)
        if current_cal_start_time is not None:
                
            new_data_to_append = {
                    'cal_start_date_time': [current_cal_start_time],
                    'BLC_V' : [cal_tmp_df.iloc[-1, cal_tmp_df.columns.get_loc('BLC_0_flag')]],
                    'conversion_efficiency' : [conversion_efficiency]    
                }
            new_data_to_append_df = pd.DataFrame(new_data_to_append)
            
            new_data_to_append_df.to_csv(file_path, mode='a', header=False, index=False, sep=',')

        if plot:

            # --- First Plot ---
            # Plot on the first subplot (axs[0])
            axs[cal-1].plot(cal_tmp_df.index, cal_tmp_df['BLC_Cal_sig_diff_cts_ref_norm'])
            axs[cal-1].set_title(f'BLC cal {cal}')

            # Adjust the layout to prevent titles and labels from overlapping
            plt.tight_layout()
    
    # Display the combined figure
    plt.show()



#def gen_MRs():
    #load data
    #ref normalise
    #flag
    #zero correct
    #analyse cals
    #analyse blc cals
    #apply all cal factors
    #save new files with MRs in 



