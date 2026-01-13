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
            bin_data_df = pd.DataFrame.from_dict(bin_data_dict)
            if bin_data_df['time_ms'][0] < 10000:
                bin_df.loc[i, 'is_time_reset'] = True
            # first_ms = bin_data_dict['time_ms'][0]
            # if first_ms < 0:
            #     first_ms += 4294967296      
            # if first_ms < 10000:
            #     bin_df.loc[i, 'is_time_reset'] = True
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
 
    bin_time_arr = \
        (bin_data_dict['time_ms'] / 1000) + log_start_datetime_seconds
        
    HK_start_ind = find_min_ind(bin_time_arr[0], HK_data['Time_s']) 
    
    HK_end_ind = find_min_ind(bin_time_arr[-1], HK_data['Time_s'])
    
    if HK_start_ind == HK_end_ind:
        print('\nThe log_start_datetime is incorrect')
        print(bin_time_arr[0])
        sys.exit(1)
        
    return bin_time_arr, HK_start_ind, HK_end_ind 
    
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
    processing_variables = pd.read_csv(os.path.join(
        data_dir, 'processing_variables.txt')
        )
    this_file_index = processing_variables[
        processing_variables['bin_filename'] == file].index[0]
    
    # Shift relevant columns as needed
    for channel in channel_format:
        if channel not in ['sig_counts', 'ref_counts', 'seed_LD_current']:
            continue
        
        shift_column = f'{channel}_shift'
        if shift_column in processing_variables.columns:
            #12/01/2026 changes!
            shift_value = int(float(processing_variables[shift_column][this_file_index]))
        else:
            shift_value = 0
            
        data_array = bin_data_dict[channel]
        original_length = len(data_array)
        
        if shift_value == 0:
            continue
        elif shift_value > 0:
            #12/01/2026: making them astype(float); test!
        # Positive shift (data moves right): Pad the beginning (left) with NaNs.
            pad_width = shift_value
            shifted_array = np.pad(
                data_array.astype(float), 
                [pad_width, 0], 
                mode='constant', 
                constant_values=np.nan
            )
            # Slice off the end to maintain original length
            bin_data_dict[channel] = shifted_array[:original_length]
       
        elif shift_value < 0:
        # Negative shift (data moves left): Pad the end (right) with NaNs.
            abs_shift = int(abs(shift_value))
            pad_width = (0, abs_shift)
            shifted_array = np.pad(
                data_array.astype(float), 
                pad_width, 
                mode='constant', 
                constant_values=np.nan
            )
            # Slice off the beginning to maintain original length
            bin_data_dict[channel] = shifted_array[abs_shift:]   
            #bin_data_dict[channel] = bin_data_dict[channel].astype(int)
    
    
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
        bin_data, channel_format, channel_count
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


"""
This section contains all of the sub-functions that are used in the 
interpretation of counts data to give mixing ratios.

"""

def read_processed_files(data_dir, day_folders):
    """
    Reads, concatenates, and cleans all processed data files (.txt) generated
    by the processing pipeline within the specified day folders.
    
    The function iterates through the 'LIFProcessed_{day}' subdirectory for each
    day, reads files starting with '20' and ending with '.txt' (which are the
    final processed data files), and combines them into a single pandas DataFrame.
    It then performs data cleaning and time conversion.
    
    Parameters
    ----------
    data_dir : str
        The root directory containing the daily data subdirectories.
    day_folders : list of str
        A list of subdirectory names (e.g., '20240115') to search for processed data.
    
    Returns
    -------
    pandas.DataFrame
        A single, concatenated and cleaned DataFrame containing the data from all
        processed files. The DataFrame is sorted by the 'Date_time' column.
        Returns an empty DataFrame if no files are found or processed.
    
    Notes
    -----
    - The function assumes the processed data files are CSV/space-delimited files
      with the actual data headers starting on the **8th line (header=7)**,
      after the metadata block.
    - Missing values, represented by **-9999**, are replaced with NaN and then
      dropped, ensuring only complete rows are kept.
    - The time column is converted from **Mac time (seconds since 1904-01-01)**
      to standard pandas datetime objects.
    """
    
    print('\nreading Processed files:')
    
    dfs = []
    
    for day in day_folders:
        
        file_list = []
        processed_dir = os.path.join(data_dir, day, f"LIFProcessed_{day}")
        if not os.path.isdir(processed_dir):
            print(f"Warning: Directory not found for day {day}: {processed_dir}")
            continue
        
        # TODO check if need to sort os.listdir results
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
    """
    Applies the reference cell normalisation step.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the processed data, which must include
        the normalized difference counts for the reference channel ('ref_diff_cts_norm')
        and the normalized difference counts for all specified signal channels
        (e.g., 'H2O_diff_cts_norm').
    channels : list of str
        A list of channel names (e.g., ['H2O', 'O2']) to apply the reference
        normalization to.
    
    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with new columns appended for the
        reference-normalized difference counts. The new column names follow the
        format: '{channel}_diff_cts_ref_norm'.
    """
    
    print('\nnormalising counts to the reference cell')
    
    data = data.copy()
    
    for channel in channels:
        
        data[f'{channel}_diff_cts_ref_norm'] = data[f'{channel}_diff_cts_norm'] / data['ref_diff_cts_norm']


    data['Date_time'] = pd.to_datetime(data['Date_time'])
    fig, ref_check = plt.subplots(1,1)
    ref_check.plot(data['Date_time'], data['ref_diff_cts_norm'])
    plt.ylabel("ref_diff_cts_norm", fontsize = 24)
    plt.xlabel("Datetime", fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    
    return data


def time_frame(cts_data_ref_norm, start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    cts_data_ref_norm_time_sectioned = cts_data_ref_norm.copy()
    
    cts_data_ref_norm_time_sectioned['Date_time'] = pd.to_datetime(cts_data_ref_norm_time_sectioned['Date_time'])
    
    cts_data_ref_norm_time_sectioned = cts_data_ref_norm_time_sectioned[
        (cts_data_ref_norm_time_sectioned['Date_time'] >= start_dt) & 
        (cts_data_ref_norm_time_sectioned['Date_time'] <= end_dt)]
    
    fig, ref_check = plt.subplots(1, 1, figsize=(12, 6))
    ref_check.plot(
        cts_data_ref_norm_time_sectioned['Date_time'], 
        cts_data_ref_norm_time_sectioned['ref_diff_cts_norm'])
    
    plt.ylabel("ref_diff_counts_norm", fontsize=24)
    plt.xlabel("Datetime", fontsize=24)
    
    plt.xticks(fontsize=24) 
    plt.yticks(fontsize=24)
    plt.tight_layout()
    
    return cts_data_ref_norm_time_sectioned

def set_flags(data, pre_TS, post_TS, pre_PF, post_PF, ref_cts_limit):
    """
    Sets two distinct types of flagging mechanisms within the processed data:
    a 'Peak_find_flag' to mask periods of low reference counts, and a time-based
    flag within the 'Task' column to mark periods around task switches.
    
    The function operates in two main phases:
    
    1.  **Peak Find Flagging:** Identifies periods where the normalized reference
        counts ('ref_diff_cts_norm') fall below a specified limit. It then sets
        the 'Peak_find_flag' to 1 for a window of time surrounding these events
        to exclude potentially noisy data from the laser falling off the peak.
    
    2.  **Task Switch Flagging:** Identifies transitions in the 'Task' column
        (indicating system operation mode changes) and sets the 'Task' value to
        '8' for a defined window of time around each switch, effectively flagging
        these transition periods as invalid for analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the processed count data, including the
        'Task' column and 'ref_diff_cts_norm'.
    pre_TS : int
        The number of data points (time steps) *before* a task switch to flag
        (i.e., set 'Task' to 8).
    post_TS : int
        The number of data points (time steps) *after* a task switch to flag
        (i.e., set 'Task' to 8).
    pre_PF : int
        The number of data points (time steps) *before* a low reference count
        event to set the 'Peak_find_flag' to 1.
    post_PF : int
        The number of data points (time steps) *after* a low reference count
        event to set the 'Peak_find_flag' to 1.
    ref_cts_limit : float or int
        The threshold value for 'ref_diff_cts_norm'. Data points below this limit
        trigger the 'Peak_find_flag'.
    
    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with the following changes:
        - The new column 'Peak_find_flag' (0 or 1) is added.
        - The 'Task' column is updated to 8 during task transition periods.
        - The 'Date_time' column is set as the DataFrame index.
        - The temporary 'Task_Change' column is dropped.
    """
    print('\nsetting flags for transient data periods')
    
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
    """
    Applies a single, constant average zero-offset correction to the
    reference-normalized count data for specified channels.
    
    The correction value is calculated by:
    1. Identifying all data periods where the 'Task' is 4 (zeroing task) and the
       'Peak_find_flag' is 0 (valid data).
    2. Grouping these zero periods sequentially.
    3. Calculating the mean of the reference-normalized counts for each individual
       zero period.
    4. Applying a 3-sigma (3 * standard deviation) spike filter to exclude
       outlying zero period means.
    5. Calculating the final grand average of the filtered zero means.
    6. Subtracting this constant grand average from the entire data column.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the processed, reference-normalized count
        data and the 'Task' and 'Peak_find_flag' columns.
    channels : list of str
        A list of channel names (e.g., ['H2O', 'O2']) to which the zero correction
        will be applied.
    plot : bool, optional
        If True, generates diagnostic plots: an errorbar plot of the zero period
        means over time, and a histogram of the zero means. Default is False.
    
    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with two new columns for each channel:
        - '{channel}_zero_offset': The constant zero correction value applied.
        - '{channel}_diff_cts_ref_norm_zero_corr': The final zero-corrected data.
    
    Notes
    -----
    - For the 'sig_B' channel (if present in `channels`), an additional filter
      is applied, setting data to NaN if either 'BLC_0_flag' or 'BLC_1_flag'
      is not equal to 1.0 during the zeroing task.
    - The zero correction is **time-independent** and uses the overall mean of
      the filtered zero-period averages.
    """
    
    print('\nCalculating zero correction')
    
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

def cal_single_point(data, channels, plot=False):
    
    print('\nChecking max cal point')
    
    mfc_rounded = data['Cal_SO2_MFC_Read'].round()
    # use a mask to select all of the zero data associated with task 4
    # set the index to Date_time for averaging later
    cal_mask = (data['Task'] == 2) & (mfc_rounded == 5) & (data['Peak_find_flag'] == 0)
    # 2. Identify unique calibration events BEFORE filtering
    # This ensures every row in 'data' gets assigned to a group ID properly
    start_of_cal = (cal_mask) & (cal_mask.shift(1) != True)
    data['cal_number'] = start_of_cal.cumsum()

    # 3. Now filter the data for the actual calculation
    cts_data_cal = data[cal_mask].copy()
    grouped_cals = cts_data_cal.groupby('cal_number')
    
    if 'sig_B' in channels:
        
        cts_data_cal['sig_B_diff_cts_ref_norm'] = np.where(
            (cts_data_cal['BLC_0_flag'] == 1.0) & (cts_data_cal['BLC_1_flag'] == 1.0)
            , cts_data_cal['sig_B_diff_cts_ref_norm']
            , np.nan
            )
    
    for channel in channels:

        column_name = f'{channel}_diff_cts_ref_norm'
        
        cal_stats = grouped_cals.agg({
            column_name: ['mean', 'std', 'count'],
            'Date_time': 'mean'
            }).dropna()
        
        # Flattening the MultiIndex columns
        cal_stats.columns = ['_'.join(col).strip() for col in cal_stats.columns.values]
        
        # Renaming for clarity
        cal_stats = cal_stats.rename(columns={
            'Date_time_mean': 'cal_midpoint'
        })
        
        mean_cal = cal_stats[f'{column_name}_mean'].mean()
        std_cal = cal_stats[f'{column_name}_mean'].std()
        lower_limit = mean_cal - 3*std_cal
        upper_limit = mean_cal + 3*std_cal
        spike_mask = ((cal_stats[f'{column_name}_mean'] > lower_limit)
            & (cal_stats[f'{column_name}_mean'] < upper_limit))
        
        cal_stats = cal_stats[spike_mask]
        
        mean_cal_spikes_removed = cal_stats[f'{column_name}_mean'].mean()
        print(f'\n{channel}:\nmean cal before spike removal = {mean_cal}'
              f'\nmean cal after spike removal = {mean_cal_spikes_removed}')
        
        mean_correction = mean_cal_spikes_removed # The overall mean after spike removal
        correction_values = np.full(len(data), mean_correction)
        
        data[f'{channel}_cal_offset'] = correction_values
        #data[f'{channel}_diff_cts_ref_norm_cal_corr'] = data[column_name] - correction_values
        #print(f'cal correction applied to {channel}')
        
        if plot:
            
            fig, ax = plt.subplots( 1, 1, figsize=(12, 8))
            
            ax.errorbar(cal_stats['cal_midpoint']
                             , cal_stats[f'{column_name}_mean']
                             , yerr=cal_stats[f'{column_name}_std']
                             , linestyle=''
                             , marker='o'
                             , markersize=2
                             , capsize=2
                             , label='cal measurement means, +/- 1std'
                             )
            #ax.plot(cts_data_zero['Date_time'], cts_data_zero[column_name], label='raw zero data')
            ax.plot(data['Date_time'], data[f'{channel}_cal_offset'], label='cal correction')
            ax.set_xlabel('Date_time')
            ax.set_ylabel(column_name)
            ax.set_title(f'{channel} cal correction')
            ax.legend()
            plt.show()

    return data   

def analyse_cals(data, data_dir, channels, molecule, cal_cylinder_conc
                 , plot, save_csv, cal_task, threshold):

    print('\nsearching dataset for calibration periods')

    cts_data = data.copy()

    # Find all flow columns and sum them to a total flow
    flows = [column for column in cts_data.columns
             if 'Flow' in column]
    cts_data['total_flow'] = cts_data[flows].sum(axis=1)
    # Calculate the MR associated with the cal gas and flow
    cts_data[f'{molecule}_mr'] = (
        cts_data[f'Cal_{molecule}_MFC_Read'] / (cts_data['total_flow']
        ) * cal_cylinder_conc)

    # Isolate the columns of data required for the cal analysis
    diff_cts_columns = [f'{channel}_diff_cts_ref_norm_zero_corr'
                        for channel in channels]
    additional_columns = ['Date_time', 'Task', 'lsr_pwr_mW',
                          f'{molecule}_mr', f'Cal_{molecule}_MFC_Read',
                          'ZA_SB_MFC_Read', f'Cal_{molecule}_MFC_set']
    all_columns = diff_cts_columns + additional_columns
    # Copy the dataframe containing only the columns of interest
    cts_data = cts_data[all_columns].copy()

    # Find start of calibration periods (Task 5 starts)
    cts_data['start_of_cal'] = np.where(
        (cts_data['Task'] == cal_task) & (cts_data['Task'].shift(1) != cal_task)
        , 1
        , 0
        )
    # Mask for calibration periods with cal SB off
    cal_task_mask = ((cts_data['Task'] == cal_task))# &
                     #(cts_data['ZA_SB_MFC_Read'] <= 0.47))
    cts_data = cts_data[cal_task_mask]
    # Cumulatively sum the start_of_cal flags to number the calibrations
    cts_data['cal_number'] = cts_data['start_of_cal'].cumsum()

    num_cals = cts_data['cal_number'].max()
    print(f'\n{num_cals} calibrations found')

    # --- PLOTTING SETUP ---
    if plot:
        ROWS = 4
        COLS = 8
        MAX_PLOTS = ROWS * COLS
        plot_counter = 0
        fig = None
        ax = None
    # ----------------------

    fieldnames = ['cal_number', 'cal_start_date_time', 'avg_lsr_pwr', 'slope',
                  'intercept', 'R2', 'slope_std_err']

    for channel in channels:

        print(f'\nAnalysing cals in {channel}:')

        if save_csv:
            file_path = os.path.join(data_dir, f'{channel}_cal_data.txt')
            # Open the file for writing (or append if it exists)
            txtfile = open(file_path, 'w', newline='')
            writer = csv.DictWriter(txtfile, fieldnames=fieldnames)

            writer.writeheader()
            
        else:
            # Create dummy objects if not making CSV
            writer = None
            txtfile = None

        try:
            cts_data_cal = cts_data.copy()
            cals = cts_data_cal.groupby('cal_number')

            for cal_num, cal_df in cals:

                # --- PLOTTING LOGIC START ---
                if plot:
                    if plot_counter % MAX_PLOTS == 0:
                        if fig is not None:
                            plt.tight_layout()
                            plt.show()

                        fig, ax = plt.subplots(ROWS, COLS,
                                               figsize=(25, 12))
                        fig.suptitle(f'Trimmed Calibration Analysis: '
                                     f'Time-Series & Regression ({channel})',
                                     fontsize=16)
                        ax = ax.flatten()

                    # Two plots per cal: Time-series and Regression
                    current_ax_reg = ax[plot_counter % MAX_PLOTS]
                    current_ax_time = ax[(plot_counter % MAX_PLOTS) + 1]
                # --- PLOTTING LOGIC END ---

                cal_df = cal_df.reset_index(drop=True).copy(deep=True)

                if len(cal_df) == 0:
                    continue

                print(f'\rcal {cal_num}', end='')

                cal_start_time = cal_df['Date_time'][0]

                # Identify individual calibration steps (points)
                cal_df['cal_point_switch'] = np.where(
                    cal_df[f'Cal_{molecule}_MFC_set'] !=
                    cal_df[f'Cal_{molecule}_MFC_set'].shift(1)
                    , 1
                    , 0
                    )
                cal_df['cal_point'] = cal_df['cal_point_switch'].cumsum()
                cal_points = cal_df.groupby('cal_point')

                # Reset steady_state column
                cal_df['stable_cal_point'] = False

                # ==========================================================
                # --- CORRECTED STEADY-STATE (TRIM) LOGIC ---
                # This applies a 5% trim to the start and end of EACH cal_point
                # ==========================================================
                for cal_point, cal_point_df in cal_points:

                    point_length = len(cal_point_df)

                    # Calculate the number of points to trim (5% of length)
                    trim_n = int(np.ceil(point_length * 0.05))

                    # Check if there's enough data left (> 10% trimmed)
                    if point_length > 2 * trim_n:

                        # Get indices of the middle 90% (trimmed data)
                        stable_cal_point_indices = cal_point_df.iloc[
                            trim_n : point_length - trim_n].index

                        # Set 'stable_cal_point' to True for these indices
                        cal_df.loc[stable_cal_point_indices,
                                   'stable_cal_point'] = True
                # ==========================================================

                stable_cal_point_mask = cal_df['stable_cal_point'] == True
                cal_df_filtered = cal_df[stable_cal_point_mask].copy()

                regression_cols = [f'{molecule}_mr',
                                   f'{channel}_diff_cts_ref_norm_zero_corr']
                cal_df_cleaned = cal_df_filtered.replace(
                    [np.inf, -np.inf], np.nan).dropna(
                        subset=regression_cols)

                # --- Prepare data for regression ---
                X = cal_df_cleaned[f'{molecule}_mr']
                Y = cal_df_cleaned[f'{channel}_diff_cts_ref_norm_zero_corr']

                cal_laser_power = cal_df_filtered['lsr_pwr_mW'].mean()

                # Check for sufficient data points before regression
                if len(X) < 2 or X.nunique() < 2:
                    print(' filtered data has no points')
                    slope, intercept, r_value, p_value, \
                        std_err_of_slope = [np.nan] * 5
                else:
                    slope, intercept, r_value, p_value, \
                        std_err_of_slope = linregress(X, Y)

                if plot:
                    # ==========================================================
                    # --- PLOT 1: TIME SERIES ---
                    # ==========================================================

                    # 1. Plot all data points against time
                    current_ax_time.plot(cal_df['Date_time'],
                                         cal_df[f'{channel}_diff_cts_ref_norm_zero_corr'],
                                         label='All Data', color='gray',
                                         linewidth=1, alpha=0.5)

                    # 2. Highlight the steady-state regions
                    current_ax_time.plot(
                        cal_df_filtered['Date_time'],
                        cal_df_filtered[f'{channel}_diff_cts_ref_norm_zero_corr'],
                        label='Trimmed (90%)', color='firebrick',
                        linewidth=1)

                    # Formatting for Time Plot
                    current_ax_time.set_title(
                        f'Cal {cal_num} - Time Series (90% Trim)',
                        fontsize=8)
                    current_ax_time.tick_params(axis='both', which='major',
                                                labelsize=6)
                    current_ax_time.tick_params(axis='x', rotation=45)
                    current_ax_time.set_ylabel('Signal (norm. cts)',
                                               fontsize=7)
                    current_ax_time.legend(loc='upper right', fontsize=6)

                    # ==========================================================
                    # --- PLOT 2: REGRESSION ---
                    # ==========================================================

                    # 1. Plot all data points for this cal (as background)
                    current_ax_reg.scatter(
                        cal_df[f'{molecule}_mr'],
                        cal_df[f'{channel}_diff_cts_ref_norm_zero_corr'],
                        label='All Data', s=5, alpha=0.3, color='gray')

                    # 2. Plot filtered (trimmed) points
                    current_ax_reg.scatter(X, Y,
                                           label='Trimmed Data', s=10,
                                           color='darkslateblue')

                    # 3. Plot the regression line if successful
                    if not np.isnan(slope):
                        x_fit = np.linspace(X.min(), X.max(), 100)
                        y_fit = slope * x_fit + intercept
                        current_ax_reg.plot(
                            x_fit, y_fit,
                            label=f'Fit (R2: {r_value**2:.2f})',
                            color='red', linestyle='--')
                        current_ax_reg.text(
                            0.05, 0.95, f'Slope: {slope:.2e}',
                            transform=current_ax_reg.transAxes,
                            verticalalignment='top', fontsize=6)
                    else:
                        current_ax_reg.text(
                            0.5, 0.5, 'Regression analysis failed',
                            transform=current_ax_reg.transAxes,
                            verticalalignment='center',
                            horizontalalignment='center', color='red')

                    # Formatting for Regression Plot
                    current_ax_reg.set_title(
                        f'Cal {cal_num} - Regression', fontsize=8)
                    current_ax_reg.tick_params(axis='both', which='major',
                                                labelsize=6)
                    current_ax_reg.set_xlabel('Mixing Ratio (MR)',
                                              fontsize=7)
                    current_ax_reg.legend(loc='lower right', fontsize=6)

                    plot_counter += 2


                new_data_to_append = {
                    'cal_number': cal_num,
                    'cal_start_date_time': cal_start_time,
                    'avg_lsr_pwr': cal_laser_power,
                    'slope': slope,
                    'intercept': intercept,
                    'R2': r_value**2,
                    'slope_std_err': std_err_of_slope
                    }

                if save_csv:
                    writer.writerow(new_data_to_append)

            # After the channel loop, close the file if it was opened
            if save_csv:
                txtfile.close()

        except Exception as e:
            print(f"An error occurred for channel {channel}: {e}")
            if save_csv and txtfile:
                txtfile.close()
            continue

        # After the loop finishes, show the last partially filled figure
        if plot and fig is not None:
            for i in range(plot_counter % MAX_PLOTS, MAX_PLOTS):
                ax[i].axis('off')
            plt.tight_layout()
            plt.show()
            plot_counter = 0
            fig = None
            ax = None
            
    if plot:

        # Create subplots: one row per channel
        fig, ax = plt.subplots(len(channels), 1, 
                               figsize=(12, len(channels) * 6))

        # Ensure ax is always iterable even for a single channel
        if len(channels) == 1:
            ax = [ax] 

        for i, channel in enumerate(channels):

            file_path = os.path.join(data_dir, f'{channel}_cal_data.txt')
            cal_data_df = pd.read_csv(file_path)

            # Filter out poor regressions (R2 < 0.75) and sort by time
            cal_df_filtered = cal_data_df[
                cal_data_df['R2'] >= 0.9
            ].sort_values(by='cal_start_date_time')

            # Convert time column to datetime objects
            time_data = pd.to_datetime(
                cal_df_filtered['cal_start_date_time'])
            
            # --- Plotting ---
            
            ax[i].errorbar(
                time_data,                           # X-axis: Time
                cal_df_filtered['slope'],            # Y-axis: Calibration Factor
                yerr=cal_df_filtered['slope_std_err'], # Error bars (y-uncertainty)
                fmt='o',                             # Format: 'o' for circles (scatter)
                capsize=3,                           # Size of the error bar caps
                color='darkslateblue',
                label='calculated cal factors'
            )
                               
            # --- Formatting ---
            ax[i].set_xlabel('cal start time')
            ax[i].set_ylabel('calibration factor')
            ax[i].set_title(f'{channel}')
            ax[i].legend()

        plt.tight_layout()
        plt.show()
    # --- Summary Printout ---
    print("\n" + "="*30)
    print("CALIBRATION QUALITY SUMMARY")
    print("="*30)
    
    threshold = threshold
    for channel in channels:
        file_path = os.path.join(data_dir, f'{channel}_cal_data.txt')
        if os.path.exists(file_path):
            summary_df = pd.read_csv(file_path)
            # Count total vs high quality
            total_cals = len(summary_df)
            high_quality_cals = len(summary_df[summary_df['R2'] > threshold])
            
            print(f"Channel: {channel}")
            print(f"  - Total calibrations analyzed: {total_cals}")
            print(f"  - Cals with R2 > {threshold}: {high_quality_cals} ({(high_quality_cals/total_cals*100):.1f}%)")
        else:
            print(f"Channel: {channel} - No calibration file found.")
    print("="*30 + "\n")   
    return
        
def analyse_BLC_cals(data, data_dir, plot, save_csv, BLC_cal_task): 
    
    print('\nsearching dataset for BLC calibration periods')
    
    cts_data = data[
        ['sig_B_diff_cts_ref_norm_zero_corr', 'Task', 'Date_time'
        , 'BLC_0_flag', 'BLC_1_flag', 'lsr_pwr_mW', 'Cal_NO_MFC_Read']
        ].copy()
    
    cts_data['start_of_BLC_cal'] = np.where(
        (cts_data['Task'] == BLC_cal_task) & 
        (cts_data['Task'].shift(1) != BLC_cal_task)
        , 1
        , 0
        )
    
    cal_task_mask = (cts_data['Task'] == BLC_cal_task)
    cts_data = cts_data[cal_task_mask]
    
    cts_data['BLC_cal_number'] = cts_data['start_of_BLC_cal'].cumsum()
    
    num_BLC_cals = cts_data['BLC_cal_number'].max()
    print(f'\n{num_BLC_cals} BLC calibrations found')
    
    # --- PLOTTING SETUP ---
    if plot:
        ROWS = 4
        COLS = 4
        MAX_PLOTS = ROWS * COLS
        plot_counter = 0
        fig = None
        ax = None
    # ----------------------
    
    fieldnames = ['cal_num', 'cal_start_date_time', 'avg_lsr_pwr'
                   ,'BLC_0_v', 'BLC_1_v','conversion_efficiency',]
    
    print('\nAnalysing BLC cals:')
    
    if save_csv:
        file_path = os.path.join(data_dir, 'BLC_cal_data.txt')
        # Open the file for writing (or append if it exists)
        txtfile = open(file_path, 'w', newline='')
        writer = csv.DictWriter(txtfile, fieldnames=fieldnames)

        writer.writeheader()
        
    else:
        # Create dummy objects if not making CSV
        writer = None
        txtfile = None
        
    
    cts_data_cal = cts_data.copy()
    BLC_cals = cts_data_cal.groupby('BLC_cal_number')
    
    
    for cal_num, cal_df in BLC_cals:
        
        # --- PLOTTING LOGIC START (Setup) ---
        if plot:
            # Check if a new figure is needed (1 plot per cal)
            if plot_counter % MAX_PLOTS == 0:
                if fig is not None:
                    plt.tight_layout()
                    plt.show()

                # Create a new figure
                fig, ax = plt.subplots(ROWS, COLS,
                                       figsize=(25, 12))
                fig.suptitle('BLC Calibration Analysis: Time-Series with Means',
                             fontsize=16)
                ax = ax.flatten()

            current_ax = ax[plot_counter % MAX_PLOTS]
        # --- PLOTTING LOGIC END (Setup) ---
        
        print(f'\rBLC cal {cal_num}', end='')
        
        cal_start_time = cal_df['Date_time'].iloc[0]
        laser_power = cal_df['lsr_pwr_mW'].mean()
        
        
        cal_df = cal_df.reset_index(drop=True)
        cal_length = len(cal_df)
        split_points = [
            0
            , np.round(0.25*cal_length)
            , np.round(0.5*cal_length)
            , np.round(0.75*cal_length)
            ]
        cal_df['split_points'] = np.where(
            cal_df.index.isin(split_points)
            , 1
            , 0
            )
        cal_df['BLC_cal_quarter'] = cal_df['split_points'].cumsum()
        cal_points = cal_df.groupby('BLC_cal_quarter')
        
        BLC_on = cal_df['BLC_cal_quarter'].isin([2, 4])
        BLC_0_v = cal_df[BLC_on]['BLC_0_flag'].mean()
        BLC_1_v = cal_df[BLC_on]['BLC_1_flag'].mean()
        
        quarter_means = []
        quarter_data_list = []
        
        for cal_point, cal_point_df in cal_points:
            
            cal_point_df = cal_point_df.dropna(subset = ['sig_B_diff_cts_ref_norm_zero_corr'])
            
            point_length = len(cal_point_df)
            trim_n = int(np.ceil(point_length * 0.1))
                
            stable_cal_point_indices = cal_point_df.iloc[
                trim_n : point_length - trim_n].index
            
            mask = cal_point_df.index.isin(stable_cal_point_indices)
            
            stable_cal_point_df = cal_point_df[mask]
            
            average = stable_cal_point_df['sig_B_diff_cts_ref_norm_zero_corr'].mean()
            
            quarter_means.append(average)
            quarter_data_list.append(stable_cal_point_df)
            
        result_dict = {
            'Q1_Mean': quarter_means[0],
            'Q2_Mean': quarter_means[1],
            'Q3_Mean': quarter_means[2],
            'Q4_Mean': quarter_means[3]
            }
        
        denom = result_dict['Q1_Mean'] - result_dict['Q3_Mean']
        
        if np.isclose(denom, 0.0):
            conversion_efficiency = np.nan
        else:
            conversion_efficiency = (
                1 - ((result_dict['Q2_Mean']-result_dict['Q4_Mean']) / denom)
                )
        
        # --- PLOTTING LOGIC CONTINUED (Drawing the Plot) ---
        if plot:
            # 1. Plot ALL data (background)
            current_ax.plot(cal_df.index,
                            cal_df['sig_B_diff_cts_ref_norm_zero_corr'],
                            label='All Data', color='lightgray', linewidth=1)
            
            # 2. Plot TRIMMED data and Mean Lines
            colors = ['darkgreen', 'darkred', 'darkgreen', 'darkred']
            for i, df in enumerate(quarter_data_list):
                q_mean = quarter_means[i]
                
                # Plot stable region trace
                current_ax.plot(df.index,
                                df['sig_B_diff_cts_ref_norm_zero_corr'],
                                label=f'Q{i+1} Trimmed', linewidth=2,
                                color=colors[i], alpha=0.8, zorder=2)
                
                # Plot mean line across the stable region
                q_start = df.index.min()
                q_end = df.index.max()
                current_ax.hlines(q_mean, q_start, q_end,
                                  color='black', linestyle='--', linewidth=3, zorder=3)
                
                # Add mean value label
                current_ax.text(q_start + (q_end - q_start)/2, q_mean, f'{q_mean:.4f}',
                                fontsize=6, ha='center', va='bottom', backgroundcolor='white')
                
                
            # 3. Add Conversion Efficiency as Text
            formatted_start_time = cal_start_time.strftime('%Y-%m-%d %H:%M')
            title_text = f"Cal {cal_num} | {formatted_start_time} | Eff: {conversion_efficiency:.3f} | Pwr: {laser_power:.2f} mW"
            current_ax.set_title(title_text, fontsize=8)
            
            # Formatting
            current_ax.tick_params(axis='both', which='major', labelsize=6)
            current_ax.set_xlabel('Index Point', fontsize=7)
            current_ax.set_ylabel('Signal (norm. cts)', fontsize=7)
            
            
            # Use specific colors for the BLC on/off status if helpful
            # BLC_0_V / BLC_1_V are means, assuming they correspond to the status in the cal
            
            current_ax.text(0.95, 0.85, f'BLC 0: {BLC_0_v: .1f}V', transform=current_ax.transAxes, 
                            fontsize=7, color='blue', ha='right')
            current_ax.text(0.95, 0.75, f'BLC 1: {BLC_1_v: .1f}V', transform=current_ax.transAxes, 
                            fontsize=7, color='blue', ha='right')
            
            current_ax.tick_params(axis='x', rotation=0)

            plot_counter += 1
        # --- PLOTTING LOGIC END (Drawing the Plot) ---
        
        
        csv_row = {
            'cal_num': cal_num,
            'cal_start_date_time': cal_start_time,
            'avg_lsr_pwr': laser_power,
            'BLC_0_v': BLC_0_v,
            'BLC_1_v': BLC_1_v,
            'conversion_efficiency': conversion_efficiency
            }
        
        if save_csv:
            writer.writerow(csv_row)
        
    if save_csv:
        txtfile.close()
        
    # --- PLOT CLEANUP AFTER LOOP ---
    if plot and fig is not None:
        # Turn off any unused subplots
        for i in range(plot_counter % MAX_PLOTS, MAX_PLOTS):
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()
    
def apply_cals_average(data_dir, data, channels, BLC, R2_limit):
    
    data_MRs = data.copy()
    # Create subplots: one row per channel
    fig, ax = plt.subplots(len(channels), 1, 
                           figsize=(12, len(channels) * 6))
    
    # Ensure ax is always iterable even for a single channel
    if len(channels) == 1:
        ax = [ax]
    
    for i, (channel, molecule) in enumerate(channels.items()):

        file_path = os.path.join(data_dir, f'{channel}_cal_data.txt')
        cal_data_df = pd.read_csv(file_path)

        # Filter out poor regressions (R2 < 0.75) and sort by time
        cal_df_filtered = cal_data_df[
            cal_data_df['R2'] >= R2_limit
        ].sort_values(by='cal_start_date_time')
        
        avg_cal_factor = cal_df_filtered['slope'].mean()
        
        data_MRs[f'{channel}_cal_factor'] = avg_cal_factor
        
        data_MRs[f'amb_{molecule}_ppt'] = np.where(
            (data_MRs['Task'] == 0) & (data_MRs['Peak_find_flag'] == 0)
            , data_MRs[f'{channel}_diff_cts_ref_norm_zero_corr'] / data_MRs[f'{channel}_cal_factor']
            , np.nan
            )
    
        # Convert time column to datetime objects
        time_data = pd.to_datetime(
            cal_df_filtered['cal_start_date_time'])
        
        # --- Plotting ---
        
        ax[i].errorbar(
            time_data,                              # X-axis: Time
            cal_df_filtered['slope'],               # Y-axis: Calibration Factor
            yerr=cal_df_filtered['slope_std_err'],  # Error bars (y-uncertainty)
            fmt='o',                                # Format: 'o' for circles (scatter)
            capsize=3,                              # Size of the error bar caps
            color='darkslateblue',
            label='calculated cal factors'
        )
        ax[i].plot(
            data_MRs['Date_time'],
            data_MRs[f'{channel}_cal_factor'],
            color='firebrick'
            )
                           
        # --- Formatting ---
        ax[i].set_xlabel('cal start time')
        ax[i].set_ylabel('calibration factor')
        ax[i].set_title(f'{channel}')
        ax[i].legend()
    
    plt.tight_layout()
    plt.show()
        
    if BLC:
        
        BLC_data = pd.read_csv(os.path.join(data_dir, 'BLC_cal_data.txt'))
        mask_1V = (BLC_data['BLC_0_v'] >= 0.9) & (BLC_data['BLC_1_v'] >= 0.9)
        BLC_data_1V = BLC_data[mask_1V]
        
        avg_conv_eff = BLC_data_1V['conversion_efficiency'].mean()
        print(f'average conversion efficiency: {avg_conv_eff}')
        data_MRs['conv_eff'] = avg_conv_eff
        
        data_MRs['amb_NO2_ppt'] = (data_MRs['amb_NO2_ppt'] - data_MRs['amb_NO_ppt']) / data_MRs['conv_eff']
        
        data_MRs['amb_NO2_ppt'] = np.where(
            (data_MRs['BLC_0_flag'] >= 0.9) & (data_MRs['BLC_1_flag'] >= 0.9),
            data_MRs['amb_NO2_ppt'],
            np.nan
            )
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        time_BLC_data = pd.to_datetime(BLC_data_1V['cal_start_date_time'])
        
        ax.plot(
            time_BLC_data
            , BLC_data_1V['conversion_efficiency']
            , color='darkslateblue'
            , linestyle=''
            , marker='o'
            , label='calculated conversion efficiencies'
            )
        
        ax.plot(
            data_MRs['Date_time'],
            data_MRs['conv_eff'],
            color='firebrick'
            )
        
        ax.set_xlabel('cal start time')
        ax.set_ylabel('conversion efficiency')
        
        ax.legend()
        plt.show()
        
    columns_to_keep = ['Date_time'] + [f'amb_{value}_ppt' for value in channels.values()]
    data_MRs = data_MRs[columns_to_keep].copy()
    
    return data_MRs

def resample_data(data, averaging):
    
    resample_data = data.copy()
    resample_data = resample_data.set_index('Date_time')
    resample_data = resample_data.resample(averaging).mean()
    resample_data = resample_data.reset_index()
    
    return resample_data

def save_to_csv(data_dir, filename, data):
    
    file_path = os.path.join(data_dir, f'{filename}.txt')
    data.to_csv(file_path, mode='w', header=True, index=False, sep=',') 
    

"""
This section contains functions which are specific to certain campaigns or 
setups but which aren't needed more generally.

"""

def CARES_NO_ref_correction(data):
    print('\n\ncorrecting ref counts due to CARES saturation issue')
    
    cts_data = data.copy()
    
    epsilon = 1e-10  # A very small number close to zero

    limit = 90549
    log_arg = 1 - (cts_data['ref_diff_cts'] / limit)

    # Clip the argument for the logarithm calculation 
    # This forces all values slightly above zero, preventing a runtime warning.
    log_arg_clipped = np.clip(log_arg, a_min=epsilon, a_max=None)

    # Calculate the raw result using the clipped data
    result_raw = (182735 * (-np.log(log_arg_clipped) / 2.02)) 

    cts_data['ref_diff_cts'] = np.where(
        log_arg > 0,
        result_raw,
        np.nan
    )
    cts_data['ref_diff_cts_norm'] = (cts_data['ref_diff_cts']/cts_data['lsr_pwr_mW'])
    
    return cts_data

def CARES_NO_plot_data(data_dir, filename):
    
    print("Loading NOx data...")
    try:
        nox_file = os.path.join(data_dir, f'{filename}.txt')
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

    # --- 3. Load and Preprocess Baseline Data (Crucial for filtering) ---
    print("Loading and processing Baseline data...")
    try:
        baseline_file = os.path.join(data_dir, 'MH_G_baseComb2_2025.txt')
        baseline_data = pd.read_csv(
            baseline_file, sep=r'\s+', header=6, engine='python'
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
        NOx_data['amb_NO_ppt'], 
        np.nan
    )
    NOx_data['clean_NO2'] = np.where(
        NOx_data['B'] == 10, 
        NOx_data['amb_NO2_ppt'], 
        np.nan
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
    NOx_data['clean_NO2'] = iqr_outlier_filter(NOx_data['clean_NO2'])


    # --- 5. Resample, Aggregate, and Calculate Diurnal Medians and IQR ---
    print("Calculating 60-minute means, diurnal medians, and IQR...")
    # Set index and resample to 60-minute intervals
    NOx_data = NOx_data.set_index('Date_time')

    # Step 1: Calculate the mean concentration for every 60-minute block.
    NOx_data_60min_means = NOx_data[
        ['clean_NO', 'clean_NO2']
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
        NOx_data['amb_NO_ppt'], 
        np.nan
    )
    NOx_data['other_NO2'] = np.where(
        NOx_data['B'] != 10, 
        NOx_data['amb_NO2_ppt'], 
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
      
def join_txt(processed_data_dir, campaign):
    # 1. Gather all .txt files in the directory
    # We use sorted() to help with initial ordering, though we sort by date later to be safe.
    files = sorted([f for f in os.listdir(processed_data_dir) if f.endswith('.txt')])
    
    df_list = []
    
    for file in files:
        # Skip our output file if it happens to be in the same folder
        if file == f"v1 {campaign} data.txt": #rename as filenames? or the name its given in the script? 
            continue
            
        file_path = os.path.join(processed_data_dir, file)
        temp_df = pd.read_csv(file_path)
        df_list.append(temp_df)
    
    # 2. Stack all the files on top of each other
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Ensure the 'Date_time' column is actual dates and sort them
    combined_df['Date_time'] = pd.to_datetime(combined_df['Date_time'])
    combined_df = combined_df.sort_values(by='Date_time').reset_index(drop=True)
    
    campaign = campaign
    
    # 4. Create the full path for the new file
    output_name = f"v1_{campaign}_data.txt"
    save_path = os.path.join(processed_data_dir, output_name)
    
    # 5. Save the file (index=False ensures we don't save row numbers)
    combined_df.to_csv(save_path, index=False)
    
    print(f"Successfully created: {save_path}")
    
    return combined_df

def SO2_plot_data(processed_data_dir, filename, campaign, version, plot = True):
    print("Loading SO2 data...")
    try:
        so2_file = os.path.join(processed_data_dir, f'{filename}.txt')
        SO2_data = pd.read_csv(so2_file)
        
        SO2_data['Date_time'] = (
            pd.to_datetime(SO2_data['Date_time']).dt.floor('min')
        )
    
        campaign = campaign
        version = version
        fig, o = plt.subplots(1, 1, figsize=(10, 5))
        o.plot(SO2_data['Date_time'], SO2_data['amb_SO2_ppt'], color='blue')
        plt.title(f"SO2 {campaign} {version}")
        plt.xlabel("Time (UTC)",fontsize = 20)
        plt.ylabel("SO2 (ppt)",fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.show()
        
        return SO2_data 
    
    except FileNotFoundError:
        print(f"Error: SO2 file not found in {processed_data_dir}.")
        return None # Return None so the script doesn't crash later
    
def SO2_plot_data_GRIMSAF(processed_data_dir, filename, campaign, version, plot = True):
    print("Loading SO2 data...")
    try:
        so2_file = os.path.join(processed_data_dir, f'{filename}.txt')
        SO2_data = pd.read_csv(so2_file)
        
        SO2_data['Date_time'] = (
            pd.to_datetime(SO2_data['Date_time']).dt.floor('min')
        )
        SO2_data['Date_time'] = (
            pd.to_datetime(SO2_data['Date_time']).dt("10s")
        )
        #changed the above to make sure that it didn't use the min so we could do sec? 
        campaign = campaign
        version = version
        fig, o = plt.subplots(1, 1, figsize=(10, 5))
        o.plot(SO2_data['Date_time'], SO2_data['amb_SO2_ppt'], color='blue')
        plt.title(f"SO2 {campaign} {version}")
        plt.xlabel("Time (UTC)",fontsize = 20)
        plt.ylabel("SO2 (ppt)",fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.show()
        
        return SO2_data 
    
    except FileNotFoundError:
        print(f"Error: SO2 file not found in {processed_data_dir}.")
        return None # Return None so the script doesn't crash later
#-------------------------------------------------------------------------------#
                            #DY195 RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#

def DY195_flagged_periods(interuptions_csv_path, SO2_data):
    # 1. Load the interruptions file using the path provided
    interuptions = pd.read_csv(interuptions_csv_path)
    
    # 2. Clean up flags: fill empty with 0 and ensure they are integers
    flag_cols = ['so2_flag', 'other_flag'] 
    for col in flag_cols:
        if col in interuptions.columns:
            interuptions[col] = interuptions[col].fillna(0).astype(int)

    # 3. Ensure time columns are datetime objects
    # Use errors='coerce' just in case there's bad text in the date columns
    SO2_data['Date_time'] = pd.to_datetime(SO2_data['Date_time'])
    interuptions['Start'] = pd.to_datetime(interuptions['Start'], dayfirst=True)
    interuptions['End'] = pd.to_datetime(interuptions['End'], dayfirst=True)

    # 4. Initialize flags in SO2_data as 0
    SO2_data['so2_remove'] = 0
    SO2_data['other_remove'] = 0

    # 5. Apply the windows from the interruptions table
    for _, row in interuptions.iterrows():
        mask = (SO2_data['Date_time'] >= row['Start']) & (SO2_data['Date_time'] <= row['End'])
        
        if row['so2_flag'] == 1:
            SO2_data.loc[mask, 'so2_remove'] = 1
            
        if row['other_flag'] == 1:
            SO2_data.loc[mask, 'other_remove'] = 1

    # 6. Filter for clean data
    SO2_data_clean = SO2_data[
        (SO2_data['so2_remove'] == 0) & 
        (SO2_data['other_remove'] == 0)
    ].copy()
    plt.figure(figsize=(10, 5))
    plt.plot(SO2_data_clean["Date_time"], SO2_data_clean["amb_SO2_ppt"])
    plt.title("Cleaned SO2 Data (Flagged Periods Removed)",fontsize = 16)
    plt.ylabel("SO2 (ppt)",fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.show()
    return SO2_data_clean

def load_all_winds_DY195(folder_path):
    all_data = []
    
    # 1. Get the list of files
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"Found {len(files)} files. Starting load...")
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 2. This is the "Loader" part - it actually opens the file
            df = pd.read_csv(file_path)
            
            # 3. Basic cleaning (Dates)
            if 'time10Hz' in df.columns:
                df['time10Hz'] = pd.to_datetime(df['time10Hz'])
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    
    # 4. Combine everything
    if all_data:
        # Sort by date so the time series makes sense
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('Date_time')
        print("Success: All data combined and sorted.")
        return combined_df

def load_ship_track(underway_file_path, filtered_DY195_SO2_data):
    # 1. Use a more robust separator and remove quotes at the engine level
    df = pd.read_csv(underway_file_path, sep=None, engine='python', quotechar='"')
    
    df['time_1min'] = pd.to_datetime(df['time_1min'], dayfirst = False)
    fig, ax1 = plt.subplots(1,1)
    # Plotting COG
    ax1.plot(df['time_1min'], df["COG_deg_1min"], color="red", label="COG", alpha=0.7)
    ax1.axhspan(0, 120, facecolor='green', alpha=0.2, label='Target Range (0-120)')
    ax1.axhspan(240, 360, facecolor='green', alpha=0.2, label='Target Range (240-360)')
    ax1.set_ylim(0, 360)
    ax1.set_ylabel("Course over ground (deg)", fontsize=14)
    ax1.legend(loc='upper right')
    # Plotting SO2
    ax2 = ax1.twinx()
    filtered_DY195_SO2_data["Date_time"] = pd.to_datetime(filtered_DY195_SO2_data["Date_time"])
    ax2.plot(filtered_DY195_SO2_data["Date_time"], filtered_DY195_SO2_data["amb_SO2_ppt"], label="SO2")
    ax2.set_ylabel("SO2 (ppt)", fontsize=18)
    plt.show()
    
    return df

#need to fix!!
def DY195_in_sector(underway_data, filtered_DY195_SO2_data):
    # 1. Ensure Date_time is a column (Fixes the KeyError)
    so2 = filtered_DY195_SO2_data.copy()
    if 'Date_time' not in so2.columns:
        # If it's missing from columns, it's likely the index
        so2 = so2.reset_index()
        # Ensure the column name is exactly 'Date_time' after reset
        if 'index' in so2.columns:
            so2 = so2.rename(columns={'index': 'Date_time'})
            
    underway = underway_data.copy()
    if 'time_1min' not in underway.columns:
        underway = underway.reset_index()
    
    # 2. Standardize times and floor to minute
    underway['merge_time'] = pd.to_datetime(underway['time_1min'], dayfirst=True).dt.tz_localize(None).dt.floor('min')
    so2['merge_time'] = pd.to_datetime(so2['Date_time'], dayfirst=True).dt.tz_localize(None).dt.floor('min')
    
    # 3. Merge
    combined_df = pd.merge(underway, so2, on='merge_time', how='inner')
    
    # 4. Filter columns and apply sector logic
    combined_df = combined_df[['merge_time', 'COG_deg_1min', 'amb_SO2_ppt']]
    df_filtered = combined_df[(combined_df['COG_deg_1min'] >= 240) | 
                              (combined_df['COG_deg_1min'] <= 120)].copy()

    if df_filtered.empty:
        print("Warning: No data found within the specified COG sector.")
        return df_filtered

    # 5. REINDEXING to create gaps for the plot
    df_filtered = df_filtered.set_index('merge_time').sort_index()
    full_range = pd.date_range(start=df_filtered.index.min(), 
                               end=df_filtered.index.max(), 
                               freq='1min')
    df_reindexed = df_filtered.reindex(full_range)

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df_reindexed.index, df_reindexed["amb_SO2_ppt"], 
            color='blue', linewidth=1.5, label='In-Sector $SO_2$')
    
    ax.set_xlabel("Datetime (UTC)", fontsize=20)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=20)
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=20)
    ax.set_title("1min averaged $SO_2$ (ppt)", fontsize=20)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))    
    
    plt.tight_layout()
    plt.show() 
    
    # IMPORTANT: Reset the index here before returning so your NEXT functions don't fail!
    return df_reindexed.reset_index().rename(columns={'index': 'Date_time'})
    return df_reindexed.reset_index().rename(columns={'index': 'time_1min'})

def DY195_in_sector_5min(underway_data, SO2_1min_in_sector):
    # 1. Prepare underway data: Set index and Resample to 5 min
    # We use .first() for COG to avoid circular mean errors
    underway_data['time_1min'] = pd.to_datetime(underway_data['time_1min'])
    uw_5min = underway_data.set_index('time_1min').resample('5min').first()
    
    # 2. Prepare SO2 data: Set index and Resample to 5 min
    SO2_1min_in_sector['Date_time'] = pd.to_datetime(SO2_1min_in_sector['Date_time'])
    so2_5min = SO2_1min_in_sector.set_index('Date_time').resample('5min').mean()
    
    fig, so2_timeseries = plt.subplots(1,1)
    so2_timeseries.plot(so2_5min.index, so2_5min["amb_SO2_ppt"])
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.ylabel("$SO_2$ (ppt)", fontsize = 20)
    plt.xlabel("Datetime (UTC)", fontsize = 20)
    plt.title("5min averaged", fontsize = 20)
    
    # 3. Merge on the 5-minute timestamps
    combined_df = pd.merge(
        uw_5min[['COG_deg_1min']], 
        so2_5min[['amb_SO2_ppt']], 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    # 4. Apply the sector filter (Only keep data in the 'clean air' sector)
    mask = (combined_df['COG_deg_1min'] >= 240) | (combined_df['COG_deg_1min'] <= 120)
    df_filtered = combined_df[mask].copy()

    # 5. Reindex to a full 5-minute range to keep gaps in the plot
    if not df_filtered.empty:
        full_range = pd.date_range(start=df_filtered.index.min(), 
                                   end=df_filtered.index.max(), 
                                   freq='5min')
        df_reindexed = df_filtered.reindex(full_range)
    else:
        df_reindexed = df_filtered

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # By removing 'marker', it stays a line plot. 
    # Because of the reindexing, the line will break automatically at gaps.
    ax.plot(df_reindexed.index, 
            df_reindexed["amb_SO2_ppt"], 
            color='blue', 
            linewidth=1.5) 
 
    ax.set_xlabel("Datetime (UTC)", fontsize=20)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=20)
    ax.set_title("5min averaged SO2 (ppt)", fontsize=20)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.tight_layout()
    plt.show()
    
    return combined_df
    
    return df_reindexed.reset_index().rename(columns={'index': 'time_5min'})

def DY195_in_sector_diurnal_1min(SO2_1min_in_sector):
    #copy all the data as appropriate!
    SO2_1min_in_sector = SO2_1min_in_sector.rename(columns={'time_1min': 'Date_time'})
    SO2_1min_in_sector['Date_time'] = pd.to_datetime(SO2_1min_in_sector['Date_time'], 
                                                     utc=True).dt.tz_convert('Etc/GMT+1')
    SO2_in_sector_1_min_for_hour = SO2_1min_in_sector.copy()
    SO2_in_sector_1_min_for_10_mins = SO2_1min_in_sector.copy()
    
    
    #DOING with 1min averaged data!
    SO2_in_sector_1_min_for_hour['hour'] = SO2_in_sector_1_min_for_hour['Date_time'].dt.hour
    grouped = SO2_in_sector_1_min_for_hour.groupby('hour')['amb_SO2_ppt']
    diurnal_mean_1_1 = grouped.mean()
    diurnal_se_1_1 = grouped.sem()
    fig, SO2_1min_1hour = plt.subplots(figsize = (10,5))
    SO2_1min_1hour.plot(diurnal_mean_1_1.index, diurnal_mean_1_1.values, 
                        marker='o', linestyle='-', color='black', label='Mean SO₂')
    # Plot shaded SE band
    SO2_1min_1hour.fill_between(
        diurnal_mean_1_1.index,
        diurnal_mean_1_1 - diurnal_se_1_1,
        diurnal_mean_1_1 + diurnal_se_1_1,
        color='green', alpha=0.3, label='±1 SE'
    )
    # Styling
    plt.title("1hour diurnal (from 1min averaging)", fontsize = 20)
    plt.xlabel("Hour of day (UTC -1)", fontsize=22)
    plt.ylabel("SO₂ (ppt)", fontsize=22)
    plt.xticks(range(0, 24), fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    #import matplotlib as mpl
    #mpl.rcParams['svg.fonttype'] = 'none'
    plt.tight_layout()
    
    # --- 1min average to a 10min resolution diurnal (144 bins) ---
    SO2_in_sector_1_min_for_10_mins['time_of_day'] = SO2_in_sector_1_min_for_10_mins['Date_time'].dt.floor("10min").dt.time
    
    # 2. Filter the raw data (Removing the contamination)
    mask_to_remove = (SO2_in_sector_1_min_for_10_mins['Date_time'].dt.hour == 7) & \
                     (SO2_in_sector_1_min_for_10_mins['Date_time'].dt.minute <= 30)
    df_filtered = SO2_in_sector_1_min_for_10_mins[~mask_to_remove].copy()
    
    # 3. Group the filtered data
    grouped = df_filtered.groupby('time_of_day')['amb_SO2_ppt']
    diurnal_mean_1_10 = grouped.mean()
    diurnal_se_1_10 = grouped.sem()
    
    # --- THE FIX: Reindex to a full 24-hour day ---
    # This ensures 07:00 exists in the index even if it has no data
    full_day_range = pd.date_range("00:00", "23:50", freq="10min").time
    diurnal_mean_1_10 = diurnal_mean_1_10.reindex(full_day_range)
    diurnal_se_1_10 = diurnal_se_1_10.reindex(full_day_range)

    # 4. Prepare Plotting Coordinates
    x_vals_1_10 = np.arange(len(diurnal_mean_1_10)) # Always 144
    
    # 5. Guaranteed 24-Hour Ticks (Finding HH:00 in the full skeleton)
    tick_indices = [i for i, t in enumerate(diurnal_mean_1_10.index) if t.minute == 0]
    tick_labels = [diurnal_mean_1_10.index[i].strftime("%H:00") for i in tick_indices]

           
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 6. Plot (Using .values to handle the NaNs correctly)
    ax.plot(x_vals_1_10, diurnal_mean_1_10.values, 
            marker='o', markersize=4, linestyle='-', color='black', label='Mean $SO_2$')
    ax.fill_between(
        x_vals_1_10, 
        diurnal_mean_1_10.values - diurnal_se_1_10.values,
        diurnal_mean_1_10.values + diurnal_se_1_10.values,
        color='green', alpha=0.3, label='±1 SE'
    )
    
    # Display the combined figure
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=22)
    plt.yticks(fontsize=22)

    ax.set_title("10-minute Diurnal Cycle (Full 24h Axis)", fontsize=20)
    ax.set_xlabel("Hour of day (UTC-1)", fontsize=22)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=22) 
    ax.legend(fontsize=18)
    
    plt.tight_layout()
    plt.show()
    
def DY195_in_sector_diurnal_5min(SO2_5min_in_sector):
    # Standardize data and handle timezone
    SO2_5min_in_sector = SO2_5min_in_sector.reset_index()
    if 'Date_time' not in SO2_5min_in_sector.columns and 'index' in SO2_5min_in_sector.columns:
        SO2_5min_in_sector = SO2_5min_in_sector.rename(columns={'index': 'Date_time'})
    
    SO2_5min_in_sector['Date_time'] = pd.to_datetime(SO2_5min_in_sector['Date_time'], 
                                                    utc=True).dt.tz_convert('Etc/GMT+1')
    SO2_in_sector_5_min_for_hour = SO2_5min_in_sector.copy()
    SO2_in_sector_5_min_for_10_mins = SO2_5min_in_sector.copy()
    #DOING with 1min averaged data!
    SO2_in_sector_5_min_for_hour['hour'] =  SO2_in_sector_5_min_for_hour['Date_time'].dt.hour
    grouped = SO2_in_sector_5_min_for_hour.groupby('hour')['amb_SO2_ppt']
    diurnal_mean_5_1 = grouped.mean()
    diurnal_se_5_1 = grouped.sem()
    
    fig, SO2_5min_1hour = plt.subplots(figsize = (10,5))
    
    SO2_5min_1hour.plot(diurnal_mean_5_1.index, diurnal_mean_5_1.values, 
                        marker='o', linestyle='-', color='black', label='Mean SO₂')
    # Plot shaded SE band
    SO2_5min_1hour.fill_between(
        diurnal_mean_5_1.index,
        diurnal_mean_5_1 - diurnal_se_5_1,
        diurnal_mean_5_1 + diurnal_se_5_1,
        color='green', alpha=0.3, label='±1 SE'
    )
    # Styling
    plt.title("1hour diurnal (from 5min averaging)", fontsize = 20)
    plt.xlabel("Hour of day (UTC -1)", fontsize=22)
    plt.ylabel("SO₂ (ppt)", fontsize=22)
    plt.xticks(range(0, 24), fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    #import matplotlib as mpl
    #mpl.rcParams['svg.fonttype'] = 'none'
    plt.tight_layout()
    
    # --- 1min average to a 10min resolution diurnal (144 bins) ---
    SO2_in_sector_5_min_for_10_mins['time_of_day'] = SO2_in_sector_5_min_for_10_mins['Date_time'].dt.floor("10min").dt.time
    
    # 2. Filter the raw data (Removing the contamination)
    mask_to_remove = (SO2_in_sector_5_min_for_10_mins['Date_time'].dt.hour == 7) & \
                     (SO2_in_sector_5_min_for_10_mins['Date_time'].dt.minute <= 30)
    df_filtered = SO2_in_sector_5_min_for_10_mins[~mask_to_remove].copy()
    
    # 3. Group the filtered data
    grouped = df_filtered.groupby('time_of_day')['amb_SO2_ppt']
    diurnal_mean_5_10 = grouped.mean()
    diurnal_se_5_10 = grouped.sem()
    
    # --- THE FIX: Reindex to a full 24-hour day ---
    # This ensures 07:00 exists in the index even if it has no data
    full_day_range = pd.date_range("00:00", "23:50", freq="10min").time
    diurnal_mean_5_10 = diurnal_mean_5_10.reindex(full_day_range)
    diurnal_se_5_10 = diurnal_se_5_10.reindex(full_day_range)

    # 4. Prepare Plotting Coordinates
    x_vals_5_10 = np.arange(len(diurnal_mean_5_10)) # Always 144
    
    # 5. Guaranteed 24-Hour Ticks (Finding HH:00 in the full skeleton)
    tick_indices = [i for i, t in enumerate(diurnal_mean_5_10.index) if t.minute == 0]
    tick_labels = [diurnal_mean_5_10.index[i].strftime("%H:00") for i in tick_indices]

           
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 6. Plot (Using .values to handle the NaNs correctly)
    ax.plot(x_vals_5_10, diurnal_mean_5_10.values, 
            marker='o', markersize=4, linestyle='-', color='black', label='Mean $SO_2$')
    ax.fill_between(
        x_vals_5_10, 
        diurnal_mean_5_10.values - diurnal_se_5_10.values,
        diurnal_mean_5_10.values + diurnal_se_5_10.values,
        color='green', alpha=0.3, label='±1 SE'
    )
    
    # Display the combined figure
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=22)
    plt.yticks(fontsize=22)

    ax.set_title("10-minute Diurnal Cycle (Full 24h Axis)", fontsize=20)
    ax.set_xlabel("Hour of day (UTC-1)", fontsize=22)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=22) 
    ax.legend(fontsize=18)
    
    plt.tight_layout()
    plt.show()
    
    return None
    
def cloud_fraction(cloud_fraction_data_path, SO2_1min_in_sector):
    # 1. Load and convert cloud data
    cloud_fraction_data = pd.read_csv(cloud_fraction_data_path)
    cloud_fraction_data['time'] = pd.to_datetime(cloud_fraction_data['YYYY-MM-DD_HHMMSS'], dayfirst=False)
    
    # 2. Prepare SO2 data - Ensure it has a DatetimeIndex for resampling
    SO2_sec = SO2_1min_in_sector.copy()
    # We use the column 'time_1min' from your previous function output
    SO2_sec['Date_time'] = pd.to_datetime(SO2_sec['Date_time'])
    SO2_sec = SO2_sec.set_index('Date_time').sort_index()
    
    # 3. Resample SO2 to 30 minutes
    # This solves the TypeError by having a DatetimeIndex
    so2_30min = (SO2_sec['amb_SO2_ppt']
                 .resample("30min")
                 .mean()
                 .reset_index())
    
    # RENAME so it matches the cloud data
    so2_30min = so2_30min.rename(columns={'Date_time': 'time'})
    
    # 4. Prepare Cloud data to 30 minutes
    cloud_30min = (cloud_fraction_data
                   .set_index('time')
                   .sort_index()['Cloud_frac']
                   .resample("30min")
                   .mean()
                   .reset_index())

    # 5. Merge the two 30-minute datasets (Both now have a 'time' column)
    com_df = pd.merge(cloud_30min, so2_30min, on='time', how='inner')
    
    # 6. Reindex to show gaps (optional but good for time-series plotting)
    com_df = com_df.set_index('time')
    if not com_df.empty:
        full_range = pd.date_range(start=com_df.index.min(), 
                                   end=com_df.index.max(), 
                                   freq='30min')
        com_df = com_df.reindex(full_range)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(com_df.index, com_df["amb_SO2_ppt"], color='blue', label='$SO_2$')
    ax1.set_ylabel("$SO_2$ (ppt)", fontsize=18, color='blue')    
    # Adding a second axis for Cloud Fraction since scales are different
    ax2 = ax1.twinx()
    ax2.scatter(com_df.index, com_df["Cloud_frac"], color='gray', label='Cloud Frac')
    ax2.set_ylabel("Cloud Fraction", fontsize=18, color='gray')
    ax1.set_xlabel("Datetime (UTC)", fontsize=18)
    plt.show()
    
    fig, sector = plt.subplots(1, 1, figsize=(12, 6))
    sector.scatter(com_df['Cloud_frac'], com_df["amb_SO2_ppt"], color='blue', linewidth=1.5)
    plt.xlabel("Cloud fraction", fontsize=24)
    plt.ylabel("$SO_2$ (ppt)", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()
    
    comb_df = com_df.reset_index().rename(columns={'index': 'time'})
    comb_df['time'] = pd.to_datetime(comb_df["time"], utc=True).dt.tz_convert('Etc/GMT+1')

       # 1. Define the contamination mask (07:00 - 07:30)
    # This identifies rows where the hour is 7 AND the minute is 30 or less
    mask_to_remove = (comb_df['time'].dt.hour == 7) & (comb_df['time'].dt.minute <= 30)
    
    # 2. Apply the mask to create a clean dataset
    # The ~ symbol means "NOT", so we keep everything except the mask
    df_filtered_cloud = comb_df[~mask_to_remove].copy()
    
    # 3. Apply the High Cloud filter to the CLEANED data
    high_cloud_df = df_filtered_cloud[df_filtered_cloud['Cloud_frac'] < 0.5].copy()
    
    # 4. Extract hour and group
    high_cloud_df['hour'] = high_cloud_df['time'].dt.hour
    grouped = high_cloud_df.groupby('hour')['amb_SO2_ppt']
    
    diurnal_mean = grouped.mean().reindex(range(0, 24))
    diurnal_se = grouped.sem().reindex(range(0, 24))
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(diurnal_mean.index, diurnal_mean.values, 
            marker='o', linestyle='-', color='darkblue', 
            linewidth=2, label='Mean $SO_2$ (Cloud < 0.5)')
    
    ax.fill_between(diurnal_mean.index, 
                    diurnal_mean - diurnal_se, 
                    diurnal_mean + diurnal_se, 
                    color='blue', alpha=0.2, label='±1 SE')
    
    ax.set_title("Diurnal $SO_2$: Lower cloud (< 0.5)", fontsize=20)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=22)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=22)
    ax.set_xticks(range(0, 24))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    """
    clean_df = combined_df.dropna(subset=['Cloud_frac', 'amb_SO2_ppt'])

    if clean_df.empty:
        print("No overlapping data found to calculate correlation.")
        return

    x = clean_df['Cloud_frac']
    y = clean_df['amb_SO2_ppt']

    # 2. Calculate Linear Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2

    # 3. Create the Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter points
    ax.scatter(x, y, alpha=0.5, color='teal', label='Data points')
    
    # Regression line
    line = slope * x + intercept
    ax.plot(x, line, color='red', linewidth=2, label=f'Fit: $R^2$ = {r_squared:.3f}')

    # Formatting
    ax.set_title("Correlation: $SO_2$ Concentration vs Cloud Fraction", fontsize=16)
    ax.set_xlabel("Cloud Fraction (0 to 1)", fontsize=14)
    ax.set_ylabel("$SO_2$ (ppt)", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"Correlation Results:\n  R-squared: {r_squared:.4f}\n  P-value: {p_value:.4e}")
    """
    

    SO2_1min_in_sector['Date_time'] = pd.to_datetime(SO2_1min_in_sector['Date_time'], 
                                                    utc=True).dt.tz_convert('Etc/GMT+1')

    # Create the exclusion mask (07:00 to 07:30 inclusive)
    mask_to_remove = (SO2_1min_in_sector['Date_time'].dt.hour == 7) & \
                     (SO2_1min_in_sector['Date_time'].dt.minute <= 30)
    
    # Apply mask to create filtered datasets for BOTH plots
    df_filtered = SO2_5min_in_sector[~mask_to_remove].copy()
    
    # --- 1. HOURLY DIURNAL PLOT ---
    df_filtered['hour'] = df_filtered['Date_time'].dt.hour
    grouped_hr = df_filtered.groupby('hour')['amb_SO2_ppt']
    diurnal_mean_5_1 = grouped_hr.mean()
    diurnal_se_5_1 = grouped_hr.sem()

    # Ensure all 24 hours are represented in index for plotting
    diurnal_mean_5_1 = diurnal_mean_5_1.reindex(range(0, 24))
    diurnal_se_5_1 = diurnal_se_5_1.reindex(range(0, 24))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(diurnal_mean_5_1.index, diurnal_mean_5_1.values, 
             marker='o', linestyle='-', color='black', label='Mean SO₂')
    ax1.fill_between(
        diurnal_mean_5_1.index,
        diurnal_mean_5_1 - diurnal_se_5_1,
        diurnal_mean_5_1 + diurnal_se_5_1,
        color='green', alpha=0.3, label='±1 SE'
    )
    ax1.set_title("1-hour Diurnal Cycle (Excluded 07:00-07:30)", fontsize=20)
    ax1.set_xlabel("Hour of day (UTC-1)", fontsize=22)
    ax1.set_ylabel("SO₂ (ppt)", fontsize=22)
    ax1.set_xticks(range(0, 24))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax1.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

    return combined_df.reset_index()
#def DY195_in_sector_winds(path):
    
# def set_flags(data, pre_TS, post_TS, pre_PF, post_PF, ref_cts_limit):
#     data = data.reset_index()
#     data['Peak_find_flag'] = 0
#     data["Task_Change"] = data.Task.shift() != data.Task
#     Task_switch_lst = data.index[data.Task_Change].tolist()
#     # Create the exclusion mask (07:00 to 07:30 inclusive)
#     mask_to_remove = (SO2_5min_in_sector['Date_time'].dt.hour == 7) & \
#                      (SO2_5min_in_sector['Date_time'].dt.minute <= 30)
#     # Apply mask to create filtered datasets for BOTH plots
#     df_filtered = SO2_5min_in_sector[~mask_to_remove].copy()
#     # --- 1. HOURLY DIURNAL PLOT ---
#     df_filtered['hour'] = df_filtered['Date_time'].dt.hour
#     grouped_hr = df_filtered.groupby('hour')['amb_SO2_ppt']
#     diurnal_mean_5_1 = grouped_hr.mean()
#     diurnal_se_5_1 = grouped_hr.sem()
#     # Ensure all 24 hours are represented in index for plotting
#     diurnal_mean_5_1 = diurnal_mean_5_1.reindex(range(0, 24))
#     diurnal_se_5_1 = diurnal_se_5_1.reindex(range(0, 24))
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     ax1.plot(diurnal_mean_5_1.index, diurnal_mean_5_1.values, 
#              marker='o', linestyle='-', color='black', label='Mean SO₂')
#     ax1.fill_between(
#         diurnal_mean_5_1.index,
#         diurnal_mean_5_1 - diurnal_se_5_1,
#         diurnal_mean_5_1 + diurnal_se_5_1,
#         color='green', alpha=0.3, label='±1 SE')
#     ax1.set_title("1-hour Diurnal Cycle (Excluded 07:00-07:30)", fontsize=20)
#     ax1.set_xlabel("Hour of day (UTC-1)", fontsize=22)
#     ax1.set_ylabel("SO₂ (ppt)", fontsize=22)
#     ax1.set_xticks(range(0, 24))
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#     ax1.legend(fontsize=18)
#     plt.tight_layout()
#     plt.show()
#     for dt_pt in data.index:
#         if data['ref_diff_cts_norm'][dt_pt] < ref_cts_limit:
#             try:
#                 data.loc[dt_pt-pre_PF:dt_pt+post_PF, 'Peak_find_flag'] = 1
#             except:
#                 data.loc[:dt_pt+post_PF, 'Peak_find_flag'] = 1
#                 print('Ref filter error index '+str(dt_pt))
#         if dt_pt > 0 and dt_pt in Task_switch_lst: 
#             data.loc[dt_pt-pre_TS:dt_pt+post_TS, 'Task'] = 8
#     data.index = data['Date_time'] 
#     return(data)
#     # --- 2. 10-MINUTE DIURNAL PLOT ---
#     df_filtered['time_of_day'] = df_filtered['Date_time'].dt.floor("10min").dt.time
#     grouped_10m = df_filtered.groupby('time_of_day')['amb_SO2_ppt']
#     diurnal_mean_5_10 = grouped_10m.mean()
#     diurnal_se_5_10 = grouped_10m.sem()
#     # Reindex to a full 24-hour day to show the gap at 07:00
#     full_day_range = pd.date_range("00:00", "23:50", freq="10min").time
#     diurnal_mean_5_10 = diurnal_mean_5_10.reindex(full_day_range)
#     diurnal_se_5_10 = diurnal_se_5_10.reindex(full_day_range)
#     x_vals_5_10 = np.arange(len(diurnal_mean_5_10)) 
#     tick_indices = [i for i, t in enumerate(diurnal_mean_5_10.index) if t.minute == 0]
#     tick_labels = [diurnal_mean_5_10.index[i].strftime("%H:00") for i in tick_indices]
#     fig, ax2 = plt.subplots(figsize=(15, 6))
#     ax2.plot(x_vals_5_10, diurnal_mean_5_10.values, 
#              marker='o', markersize=4, linestyle='-', color='black', label='Mean $SO_2$')
#     ax2.fill_between(
#         x_vals_5_10, 
#         diurnal_mean_5_10.values - diurnal_se_5_10.values,
#         diurnal_mean_5_10.values + diurnal_se_5_10.values,
#         color='green', alpha=0.3, label='±1 SE'    )
#     ax2.set_xticks(tick_indices)
#     ax2.set_xticklabels(tick_labels, rotation=45, fontsize=22)
#     plt.yticks(fontsize=22)
#     ax2.set_title("10-minute Diurnal Cycle (Excluded 07:00-07:30)", fontsize=20)
#     ax2.set_xlabel("Hour of day (UTC-1)", fontsize=22)
#     ax2.set_ylabel("$SO_2$ (ppt)", fontsize=22) 
#     ax2.legend(fontsize=18)
#     plt.tight_layout()
#     plt.show() 

#-------------------------------------------------------------------------------#
                            #MACE HEAD RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#     
def MH_flagged_periods(MH_interuptions_path, SO2_data):
    # 1. Load the interruptions file using the path provided
    interuptions = pd.read_csv(MH_interuptions_path)
    
    # 2. Clean up flags: fill empty with 0 and ensure they are integers
    flag_cols = ['so2_flag', 'other_flag'] 
    for col in flag_cols:
        if col in interuptions.columns:
            interuptions[col] = interuptions[col].fillna(0).astype(int)

    # 3. Ensure time columns are datetime objects
    # Use errors='coerce' just in case there's bad text in the date columns
    SO2_data['Date_time'] = pd.to_datetime(SO2_data['Date_time'])
    interuptions['Start'] = pd.to_datetime(interuptions['Start'], dayfirst=True)
    interuptions['End'] = pd.to_datetime(interuptions['End'], dayfirst=True)

    # 4. Initialize flags in SO2_data as 0
    SO2_data['so2_remove'] = 0
    SO2_data['other_remove'] = 0

    # 5. Apply the windows from the interruptions table
    for _, row in interuptions.iterrows():
        mask = (SO2_data['Date_time'] >= row['Start']) & (SO2_data['Date_time'] <= row['End'])
        if row['so2_flag'] == 1:
            SO2_data.loc[mask, 'so2_remove'] = 1  
        if row['other_flag'] == 1:
            SO2_data.loc[mask, 'other_remove'] = 1

    # 6. Filter for clean data
    SO2_data_clean = SO2_data[
        (SO2_data['so2_remove'] == 0) & 
        (SO2_data['other_remove'] == 0)].copy()
    plt.figure(figsize=(10, 5))
    plt.plot(SO2_data_clean["Date_time"], SO2_data_clean["amb_SO2_ppt"])
    plt.title("Cleaned SO2 Data (Flagged Periods Removed)",fontsize = 16)
    plt.ylabel("SO2 (ppt)",fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.show()
    
    return SO2_data_clean

def load_baseline_flagger(data_dir_MH_base, MH_Baseline, SO2_data):
#defined from the MH shared drive
#B=0-9 Mixed    B=10-19 Baseline   B=20-29 Local
#B=30-39 Southerly   B=40-49 Polluted   B=50-59 ExcBase  
# --- Load and Preprocess Baseline Data ---
    baseline_file = os.path.join(data_dir_MH_base, 'MH_G_baseComb2_2025.txt')
    baseline_data = pd.read_csv(
            baseline_file, sep=r'\s+', header=6, engine='python')
        # datetime conversion from separate YY, MM, DD, HH, Mn columns  
        # generates a Date_time column
    time_cols = {
            'year': 'YY'
            , 'month': 'MM'
            , 'day': 'DD'
            , 'hour': 'HH'
            , 'minute': 'Mn'
            }
    baseline_data['Date_time'] = pd.to_datetime(
            baseline_data[list(time_cols.values())].rename(
                columns={v: k for k, v in time_cols.items()}),
            errors='coerce')
       
        # Prepare baseline index for alignment
        # removes nan times, sets index and sorts chronologically
    baseline_data = baseline_data.dropna(
            subset=['Date_time']).sort_values('Date_time') 
    
    SO2_data = SO2_data      
    SO2_data = SO2_data.sort_values('Date_time')
    baseline_data = baseline_data.sort_values('Date_time')

    SO2_with_flags = pd.merge_asof(
        SO2_data, 
        baseline_data[['Date_time', 'B']], 
        on='Date_time', 
        direction='backward'
    )

    # 4. Filter for only the "Baseline" flags (10-19)
    # This keeps every SO2 point that fell into a 2h window starting with flag 10-19
    SO2_filtered = SO2_with_flags[
        (SO2_with_flags['B'] >= 10) & (SO2_with_flags['B'] < 20)].copy()

    return SO2_filtered
#-------------------------------------------------------------------------------#
                            #TASMANIA RELEVANT FUNCTIONS
#-------------------------------------------------------------------------------#
#all of the files should be in UTC
def Tasmania_flagged_periods(TAS_interuptions_path, SO2_data):
    #TODO
    interuptions = pd.read_csv(TAS_interuptions_path)
    
    # 2. Clean up flags: fill empty with 0 and ensure they are integers
    flag_cols = ['so2_flag', 'other_flag'] 
    for col in flag_cols:
        if col in interuptions.columns:
            interuptions[col] = interuptions[col].fillna(0).astype(int)

    # 3. Ensure time columns are datetime objects
    # Use errors='coerce' just in case there's bad text in the date columns
    SO2_data['Date_time'] = pd.to_datetime(SO2_data['Date_time'])
    interuptions['Start'] = pd.to_datetime(interuptions['Start'], dayfirst=True)
    interuptions['End'] = pd.to_datetime(interuptions['End'], dayfirst=True)

    # 4. Initialize flags in SO2_data as 0
    SO2_data['so2_remove'] = 0
    SO2_data['other_remove'] = 0

    # 5. Apply the windows from the interruptions table
    for _, row in interuptions.iterrows():
        mask = (SO2_data['Date_time'] >= row['Start']) & (SO2_data['Date_time'] <= row['End'])
        if row['so2_flag'] == 1:
            SO2_data.loc[mask, 'so2_remove'] = 1  
        if row['other_flag'] == 1:
            SO2_data.loc[mask, 'other_remove'] = 1

    # 6. Filter for clean data
    SO2_data_clean = SO2_data[
        (SO2_data['so2_remove'] == 0) & 
        (SO2_data['other_remove'] == 0)].copy()
    plt.figure(figsize=(10, 5))
    plt.plot(SO2_data_clean["Date_time"], SO2_data_clean["amb_SO2_ppt"])
    plt.title("Cleaned SO2 Data (Flagged Periods Removed)",fontsize = 16)
    plt.ylabel("SO2 (ppt)",fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.show()
    
    return SO2_data_clean

def Tas_baseline(T_baseline_flagger_path, data):
    #need the SO2 data and the baseline files!
    data = data
    data["Date_time"] = pd.to_datetime(data["Date_time"])
    
    T_baseline_flagger = pd.read_csv(T_baseline_flagger_path)
    
    # This swaps 'datetime' for 'Date_time'
    T_baseline_flagger = T_baseline_flagger.rename(columns={'datetime': 'Date_time'})
    T_baseline_flagger["Date_time"] = pd.to_datetime(T_baseline_flagger["Date_time"])
    #plot out the baseline data across the top of the SO2 data, to show the 
    #baseline periods during the campaign.
    fig, ax = plt.subplots(figsize=(10, 6)) # figsize sets the plot size
    ax.plot(T_baseline_flagger["Date_time"], T_baseline_flagger["BL3"], label="BL3 Data", color='blue')
    ax.fill_between(
        T_baseline_flagger['Date_time'], 0, 1,
        where=(T_baseline_flagger['BL3'] == 1),
        color='red', alpha=0.2, transform=ax.get_xaxis_transform(), label='Baseline')
    plt.ylabel("Baseline", fontsize = 24)
    r = ax.twinx()
    r.plot(data["Date_time"], data["amb_SO2_ppt"], label="Ambient Corrected Data", color='red', linestyle='--')
    plt.ylabel("SO2 (ppt)", fontsize = 24)
    ax.set_xlabel("Datetime")
    ax.set_title("Baseline flagged SO2 data")
    ax.legend(loc='best')
    plt.show()# Filter by date
    #then filter out for the periods where the baseline flag ==1
    baseline_SO2 = data.set_index('Date_time').resample('1min').ffill().reset_index()
    # merged headers = Date_time, amb_SO2_ppt, BL3
    baseline_SO2["Date_time"] = pd.to_datetime(baseline_SO2["Date_time"])
    print("Merging the SO2 and baseline flagger")
    merged = pd.merge_asof(
        baseline_SO2.sort_values('Date_time'),
        T_baseline_flagger[['Date_time', 'BL3']].sort_values('Date_time'),
        on='Date_time',
        direction='nearest',
        tolerance=pd.Timedelta('1min'))
    #filter for when BL3 ==1 (which is the baselien flag)
    SO2_baseline_flagged = merged[(merged['BL3'] ==1)].copy()
    SO2_baseline_flagged["Date_time"] = pd.to_datetime(SO2_baseline_flagged["Date_time"])
    #plot out the data when we were in the baseline sector
    fig, baseline_flagged_SO2 = plt.subplots(1,1)
    baseline_flagged_SO2.plot(SO2_baseline_flagged["Date_time"], SO2_baseline_flagged["amb_SO2_ppt"], color = "green")
    plt.ylabel("SO2 (ppt)", fontsize = 24)
    plt.xlabel("Datetime", fontsize = 24)
    plt.title("SO2 in the baseline sector")
    
    print("Producing 1min avg SO2 data for the baseline sector")
    SO2_in_baseline = SO2_baseline_flagged.drop(columns=["BL3"])
    output_filename = "SO2_TAS_baseline_1min_avg.csv"
    SO2_in_baseline.to_csv(output_filename, index=False)
    
    five_min_resample = SO2_in_baseline.resample("5min").mean()
    five_min_resample["Date_time"] = pd.to_datetime(five_min_resample["Date_time"])
    
    fig, i = plt.subplots(1,1)
    i.plot(five_min_resample["Date_time"], five_min_resample["amb_SO2_ppt"])
    plt.xlabel("Datetime", fontsize = 24)
    plt.ylabel("SO2 (ppt)", fontsize = 24)
    plt.title("5min averaging of SO2 in the baseline sector")
    
    print("Producing 5min avg SO2 data for the baseline sector")
    output_filename = "SO2_TAS_baseline_5min_avg.csv"
    five_min_resample.to_csv(output_filename, index=False)
    return SO2_in_baseline

def timezone_conversion(data):
    aedt = pytz.timezone("Australia/Sydney")
    time_zone_converted_data = data.copy()
    print("Data becoming Australian")
    time_zone_converted_data['Date_time'] = pd.to_datetime(time_zone_converted_data['Date_time'], utc=True).dt.tz_convert(aedt)
    
    #pick a random bit of the data to test that it has actually shifted the data to the right timezone!
    time_conv_subset = time_zone_converted_data.loc[300:900]
    time_utc = data.loc[300:900]
    
    fig, time_corr = plt.subplots(1,1)
    time_corr.plot(time_utc["Date_time"], time_utc["amb_SO2_ppt"], 
                   color = "lightseagreen", label = "UTC")
    time_corr.plot(time_conv_subset["Date_time"], time_conv_subset["amb_SO2_ppt"], 
                   color = "purple", label = "AEDT")
    plt.legend(loc='best')
    return time_zone_converted_data

def TAS_windrose(data, met_data_dir, wind_dir):
    data = data
    met_data = pd.read_csv(met_data_dir)
    winds = pd.read_csv(wind_dir)
    
    #give Date_time and make sure they have the same column names!
    data["Date_time"] = pd.to_datetime["Date_time"]
    met_data = met_data.rename(columns={'time': 'Date_time'})
    met_data["Date_time"] = pd.to_datetime(met_data["Date_time"])
    winds = winds.rename(columns={'time': 'Date_time'})
    winds["Date_time"] = pd.to_datetime(winds["Date_time"])
    
    #assign the variables that we need
    #winds in sector are 190-280
    #wind data is 10min
    so2_ppt = data["amb_SO2_ppt"]
    wind_speed = winds["met_wspd"]
    wind_direction = winds["met_wdir"]
    
    #plot out the wind direction and the SO2 (diagnostic)
    fig, l = plt.subplots(2,1)
    l[0].plot(winds["Date_time"], wind_direction)
    l[0].set_ylabel("Wind Direction (°)", fontsize = 24)
    l[1].plot(data["Date_time"], so2_ppt)
    l[1].set_ylabel("SO2 (ppt)", fontsize = 24)
    l[1].set_xlabel("Datetime", fontsize = 24)
    
    
#need baseline flags
#need wind direction?
#need irradiance?
#need the dirunals aswell!  
   
    
    
    
    
    
    