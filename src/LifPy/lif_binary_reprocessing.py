import pandas as pd
import lif_functions as lif    
 
working_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
               'Data Analysis\\CARES\\Mace Head Binary Data Analysis')

path = (f'{working_dir}\\processing_variables.txt')

processing_variables = pd.read_csv(path)
#date_mask = processing_variables['date'] >= 20250608
#processing_variables = processing_variables[date_mask]

for i in processing_variables.index:
    
    lif.reprocess_binary_data(
        data_freq = 10 # can take 10 or 100
        , log_start_datetime = processing_variables['log_start_datetime'][i]
        , date = str(processing_variables['date'][i])
        , working_dir = ('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\'
                         'Data Analysis\\CARES\\Mace Head Binary Data Analysis')
        , HK_file_path = 'data_HK\\' + str(processing_variables['date'][i])
        , bin_file_path = 'data_bin\\' + str(processing_variables['date'][i])
        , HK_headers_dict = ['Task', 'NO_Cell_Flow', 'NO2_Cell_Flow', 'Ref_Cell_SLPM'
                             , 'Cal_NO_MFC_set', 'Cal_NO_MFC_Read', 'Cal_SB_MFC_Read'
                             , 'Ref_NO_MFC_Read', 'BLC_0_flag', 'BLC_1_flag'] 
        , channel_count = 11 # 10 = single channel, 11 = dual channel
        , channel_format = {
            'sig_A_counts': 0
            , 'ref_counts': 1
            , 'seed_LD_current': 2
            , 'laser_pwr_PT0': [3, 4]
            , 'time_ms': [7, 8]
            , 'seed_LD_mode': 9
            , 'sig_B_counts': 10
        }
        , skip_start_bin = processing_variables['skip_start_bin'][i] 
        , skip_end_bin = processing_variables['skip_end_bin'][i]  
    )

        
        
        
    
    