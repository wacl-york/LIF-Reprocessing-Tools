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
| log_start_datetime      |               | str('%d/%m/%Y %H:%M:%S') |
| bin_file_path           |               | str() |
| HK_file_path            |               | str() |
| data_freq               |               | int() |
| skip_start              |               | int() |
| skip_end                |               | int() |
| ignore_first            |               | Boolean |
| gen_diag_plots          |               | Boolean |

## Troubleshooting