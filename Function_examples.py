import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import datetime
from matplotlib.lines import Line2D

from time_stamps import index_timestamp

from file_processes import import_HK_data
from file_processes import import_binary_LIF_cts
from file_processes import calc_SO2_mix_r

HK_data = import_HK_data('HK data', skip_start=1, skip_end=2)
# this function will read the HK data files from the 'HK data' folder, but skip the first one and last two


binary_data = import_binary_LIF_cts('Binary data', skip_start=5)
# this function will read the binary data files from the 'Binary data' folder, but skip the first 5


log_datetime_str = '20/04/2022 15:26:36'
SO2_LOD = 1000

reprocess_binary_data = calc_SO2_mix_r(binary_data, log_datetime_str, sensitivity=0.6, data_freq=5)

start_cal_ind = index_timestamp('20/04/2022 17:00:00', np.array(reprocess_binary_data['Time_ms']) / 1000)
end_cal_ind = index_timestamp('20/04/2022 17:13:00', np.array(reprocess_binary_data['Time_ms']) / 1000)

plt.title('Example Calibration (1 Hz Data)')
plt.plot(reprocess_binary_data['Time_ms'][start_cal_ind: end_cal_ind]
         , reprocess_binary_data['SO2_mr'][start_cal_ind: end_cal_ind], color='steelblue')
plt.ylabel('SO$_2$ Mix Ratio / pptv')
plt.xlabel('Time / UTC')

plt.plot([min(reprocess_binary_data['Time_ms'][start_cal_ind: end_cal_ind])
             , max(reprocess_binary_data['Time_ms'][start_cal_ind: end_cal_ind])], [0, 0], color='red', ls=':')
plt.fill_between([min(reprocess_binary_data['Time_ms'][start_cal_ind: end_cal_ind])
                    , max(reprocess_binary_data['Time_ms'][start_cal_ind: end_cal_ind])]
                 , [-SO2_LOD, -SO2_LOD], [SO2_LOD, SO2_LOD]
                 , color='red', alpha=0.2)

seconds_from_epoch = (dt.strptime('20/04/2022', '%d/%m/%Y') - dt.strptime('01/01/1904', '%d/%m/%Y')).total_seconds()
xlabels = (plt.xticks()[0] / 1000) - seconds_from_epoch
new_xlabels = [str(datetime.timedelta(seconds=label)) for label in xlabels]
plt.xticks(ticks=plt.xticks()[0], labels=new_xlabels, rotation=45)

custom_lines = [Line2D([0], [0], color='steelblue', lw=2),
                Line2D([0], [0], color='red', lw=2, ls=':')]
plt.legend(custom_lines, ['10 Hz SO$_2$ Mixing Ratio', '3\u03C3 10 Hz LOD (%.0f ppbv)' % (SO2_LOD / 1000)])

plt.tight_layout()
plt.show()
