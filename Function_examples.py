from file_processes import import_HK_data
from file_processes import import_binary_LIF_cts

HK_data = import_HK_data('HK data', skip_start=1, skip_end=2)
# this function will read the HK data files from the 'HK data' folder, but skip the first one and last two

binary_data = import_binary_LIF_cts('Binary data', skip_start=5)
# this function will read the binary data files from the 'Binary data' folder, but skip the first 5


