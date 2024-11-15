# LIF data processing package

## Setup

Run the env_setup.py script to generate an environment where you can reprocess LIF data. Provide it with the file path including the name of the folder 
you want it to create, example below.

## Key functions

The reprocess_binary_data() function will load in the binary and HK data from the default folders below. The 100 Hz binary data will be decimated to 
10 Hz (default), whilst the 5 Hz HK data will be extrapolated to 10 Hz.

```
from LifPy.reprocess import reprocess_binary_data

reprocess_binary_data(
    log_start_datetime='%d/%m/%Y %H:%M:%S'
    , bin_file_path='data/bin_data'
    , HK_file_path='data/HK_data'
    , data_freq=10
    , skip_start=0
    , skip_end=0
    , ignore_first=False
    , gen_diag_plots=True
)
```
All variables in the reprocess_binary_data() function are defined with default values (excluding log_start_datetime) that rarely need to be changed. Below is a
table describing each variable and their accepted inputs.

| var name                | desc          | input type     |
| :---------------------: |:-------------:| :------------: |
| log_start_datetime      | Take this from the log file when the LIF code initiated, usually after a hard restart.                        | str('%d/%m/%Y %H:%M:%S') |
| bin_file_path           | The location of the folder where the binary data files are located, do not include the root folder structure. | str() |
| HK_file_path            | The location of the folder where the HK data files are located, do not include the root folder structure.     | str() |
| data_freq               | The frequency (Hz) at which the processed counts will be output at, default 10 Hz is the max.                 | int(<= 10)    |
| skip_start              | The number of binary files to skip from the start, note they are in date order.                               | int(< # of bin files) |
| skip_end                | The number of binary files to skip from the end, note they are in date order.                                 | int(< # of bin files) |
| ignore_first            | If True, the first online count from each set of 8 will be discarded                                          | Boolean() |
| gen_diag_plots          | If True, the code will generate the diagnostic plots defined in bin/html_plot_prefs.txt.                      | Boolean() |

## Troubleshooting