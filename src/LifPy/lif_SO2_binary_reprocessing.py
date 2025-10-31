import lif_functions as lif     

path = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
        'Data Analysis\\CARES\\Mace Head Binary Data Analysis\\'
        'processing_variables.txt'
        )

    
lif.reprocess_binary_data(
    data_freq = 10 # can take 10 or 100
    , log_start_datetime = '05/06/2025 09:22:38'
    , date = '20250605'
    , working_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                     'Data Analysis\\CARES\\SO2_code_test')
    , HK_file_path = 'data_HK'  
    , bin_file_path = 'data_bin' 
    , HK_headers_dict = ['Task', 'Cell_Flow'] 
    , channel_count = 10 # 10 = single channel, 11 = dual channel
    , channel_format = {
        'sig_counts': 0
        , 'ref_counts': 1
        , 'seed_LD_current': 2
        , 'laser_pwr_PT0': [3, 4]
        , 'time_ms': [7, 8]
        , 'seed_LD_mode': 9
    }
    , skip_start_bin = 5
    , skip_end_bin = 0
    , histograms = True
)