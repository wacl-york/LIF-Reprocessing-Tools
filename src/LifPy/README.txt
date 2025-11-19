README.txt

LIF REPROCESSING TOOLS - PHIN_NO BRANCH

The following contains instructions for how to set up the correct environment and perform binary reprocessing on LIF data using the scripts contained within the Phin_NO branch of LIF reprocessing code.

###

Download the following python scripts and save them within a common directory to run them from:
- lif_functions.py
- lif_NO_generate_processing_variables.py / lif_SO2_generate_processing_variables.py
- lif_NO_binary_reprocessing.py / lif_SO2_binary_reprocessing.py

###

The raw data files to process must be organised in the following structure:

- A single parent directory which should be pointed to at the top of each script as data_dir
- Subdirectories for each day of data collected, in the format YYYYMMDD
- Subdirectories within each day, in the formats LIFCnts_YYYYMMDD, LIFHK_YYYYMMDD, and LIFLog_YYYYMMDD (same format as the data on the cRIO).
- Individual data files stored in the appropriate subdirectory within the appropriate day.

- By running the reprocessing code, a fourth directory will be generated within each day, in the format LIFProcessed_YYYYMMDD

###

Run the files in the following order:

1. Run lif_NO_generate_processing_variables.py / lif_SO2_generate_processing_variables.py. 

(This will take up to a few minutes depending on the amount of data that is being processed. Two text files are generated within data_dir: processing_variables.txt and soft_restarts.txt. LEAVE THESE FILES HERE until the binary reprocessing has been completed. processing_variables.txt contains information about each of the binary files including the correct log_start_datetime of the instrument when the file was created and any shifts that have been identified due to issues with data misalignments within files. soft_restarts.txt is an intermediate file that is created within the file misalignment identification step. If the data to process was collected after the update to the FPGA code and you are sure that file misalignment hasn't occurred, then the call to lif.misaligned_counts() at the bottom of lif_NO_generate_processing_variables.py / lif_SO2_generate_processing_variables.py can be commented out and the rest of the code will progress without checking for misaligned data. In this case, soft_restarts.txt will not be generated.)

2. Run lif_NO_binary_reprocessing.py / lif_SO2_binary_reprocessing.py

(This takes approximately 7 mins per file or up to 8 hours per week of continuous data. The processed data files will be generated and saved within each day's data directory. Within this script, HK_headers_dict can be adjusted to include any of the variables from the HK data that you would like transferring into the processed data files. within the call to lif.reprocess_binary_data(), data_freq can take the value of 10 or 100, giving either processed cts diff values or raw 100Hz data points. Processing at 100Hz is significantly slower than 10Hz so should only be performed on small periods of data for the purpose of troubleshooting.)


