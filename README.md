# LIF data processing package

## Setup

Run the env_setup.py script to generate an environment where you can reprocess LIF data. Provide it with the file path including the name of the folder you want it to create, example below.

## Key functions

```
from LifPy.reprocess import reprocess_binary_data

reprocess(log_start_datetime='%d/%m/%Y %H:%M:%S', bin_file_path='data/bin_data', HK_file_path='data/HK_data', data_freq=10, skip_start=0, skip_end=0, ignore_first=False, gen_diag_plots=True)
```

## Troubleshooting