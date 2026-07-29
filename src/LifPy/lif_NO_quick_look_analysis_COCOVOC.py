# -*- coding: utf-8 -*-
"""
Created 05/03/2025
@author: Pete
"""

#Tasmania reprocessing 
import Calibration_analysis_functions as lif # contains all the functions that are called from this script
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def Analyse_cals(all_data, plot, max_conc, cell):
    cts_diff_v = 'sig_'+cell+'_diff_cts'
    cts_diff_refnorm_v = 'Sig_'+cell+'_ctsdiff_ref_norm'
    data = all_data[[cts_diff_v, cts_diff_refnorm_v, 'NO_mr', 'Task_y', 'Cal_NO_MFC_Read_y', 'Cal_SB_MFC_Read', 'Cal_NO_MFC_set']].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.reset_index(drop=True)
    data['Cal_Sig_ctsdiff_ref_norm'] = data[cts_diff_refnorm_v].where(data['Task_y'] == 5)
    data['Cal_true_ppt'] = data['NO_mr'].where((data.Task_y == 5) & (data['Cal_SB_MFC_Read'] < 0.01))
    data['Cal_group'] = np.nan
    cal_num = 0
    for i in data.index:
        if (i > 0 and pd.isnull(data['Cal_true_ppt'][i]) == False and pd.isnull(data['Cal_true_ppt'][i-300:i].mean())):
            cal_num+=1
        if (pd.isnull(data['Cal_true_ppt'][i]) == False and pd.isnull(data['Cal_true_ppt'][i-301:i-10].mean()) == False):
            data['Cal_group'][i] = cal_num
    Refnorm_cal_vars = {}
    std_cal_vars = {}
    for cal in range(1,cal_num):
        cal_tmp_df = data[(data['Cal_group'] ==  cal)].copy().dropna()
        cal_tmp_df['point_filter'] = 0
        cal_tmp_df['cal_flow_diff'] = cal_tmp_df['Cal_NO_MFC_set'].diff().abs()
        for cal_pt in cal_tmp_df.index:
            if (cal_tmp_df['cal_flow_diff'][cal_pt] > 0.05):
                cal_tmp_df.loc[cal_pt-1:cal_pt+10,'point_filter'] = 1
        cal_tmp_df = cal_tmp_df[(cal_tmp_df['point_filter'] == 0) & (cal_tmp_df['Cal_true_ppt'] < max_conc)]
        if cal_tmp_df.size > 100:
            X = cal_tmp_df['Cal_true_ppt'][30:].values.reshape(-1, 1)
            Y = cal_tmp_df['Cal_Sig_ctsdiff_ref_norm'][30:].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            reg = linear_regressor.fit(X, Y)
            Y_pred = linear_regressor.predict(X)
            Norm_cal_dict = {}
            Norm_cal_dict['R2'] = reg.score(X,Y)
            Norm_cal_dict['Slope'] = reg.coef_[0,0]
            Norm_cal_dict['Intercept'] = reg.intercept_[0]
            Refnorm_cal_vars[cal] =  Norm_cal_dict
            
            Y2 = cal_tmp_df[cts_diff_v][30:].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            reg2 = linear_regressor.fit(X, Y2)
            Y2_pred = linear_regressor.predict(X)
            cal_dict = {}
            cal_dict['R2'] = reg2.score(X,Y2)
            cal_dict['Slope'] = reg2.coef_[0,0]
            cal_dict['Intercept'] = reg2.intercept_[0]
            std_cal_vars[cal] = cal_dict
            
            cal_data = pd.DataFrame({'X':cal_tmp_df['Cal_true_ppt'][30:], 'Y':cal_tmp_df['Cal_Sig_ctsdiff_ref_norm'][30:]})
            cal_data.to_csv('Cell_'+cell+'_Cal_data_'+str(cal)+'.csv')
            
            if plot:
                plt.scatter(X, Y2)
                plt.plot(X, Y2_pred, color='red')
                plt.title('Standard cal '+str(cal)+' Cell '+cell)
                plt.show()
                
                plt.plot(cal_tmp_df.index,cal_tmp_df[cts_diff_v])
                plt.title('Standard cal '+str(cal)+' Cell '+cell)
                plt.show()
                
                plt.plot(cal_tmp_df.index,cal_tmp_df['Cal_Sig_ctsdiff_ref_norm'])
                plt.title('Ref norm cal '+str(cal)+' Cell '+cell)
                plt.show()
                
                plt.scatter(X, Y)
                plt.plot(X, Y_pred, color='red')
                plt.title('Ref norm cal '+str(cal)+' Cell '+cell)
                plt.show()
    
    Std_cal_summary = pd.DataFrame(std_cal_vars).transpose()
    print('Cell '+cell+' Non ref normalised cals')
    print(Std_cal_summary)
    if (100*(Std_cal_summary.Slope.std()/Std_cal_summary.Slope.mean())) < 5:
        print('Cell '+cell+' Cal slope standard deviation < 5% of mean')
    else:
        print('Cell '+cell+' Cal slope standard deviation greater than 5% of mean')

    Refnorm_cal_summary = pd.DataFrame(Refnorm_cal_vars).transpose()
    print('Cell '+cell+' Reference cell normalised cals')
    print(Refnorm_cal_summary)
    if (100*(Refnorm_cal_summary.Slope.std()/Refnorm_cal_summary.Slope.mean())) < 5:
        print('Cell '+cell+' Ref norm cal slope standard deviation < 5% of mean')
    else:
        print('Cell '+cell+' Ref norm cal slope standard deviation greater than 5% of mean')    
    
    return(Std_cal_summary, Refnorm_cal_summary)

def Diagnostic_plots(data, vars):
    for v in vars:
        plt.plot(data['Date_time'],data[v])
        plt.title(v)
        plt.show()
        
def set_flags(data, pre_TS, post_TS, pre_PF, post_PF, ref_cts_limit):
    data = data.reset_index()
    data['Peak_find_flag'] = 0
    data["Task_Change"] = data.Task_y.shift() != data.Task_y
    Task_switch_lst = data.index[data.Task_Change].tolist()

    for dt_pt in data.index:
        if data['ref_cts_diff'][dt_pt] < ref_cts_limit:
            try:
                data['Peak_find_flag'][dt_pt-pre_PF:dt_pt+post_PF] = 1
            except:
                data['Peak_find_flag'][:dt_pt+post_PF] = 1
                print('Ref filter error index '+str(dt_pt))
        if dt_pt > 0 and dt_pt in Task_switch_lst: 
            data['Task'][dt_pt-pre_TS:dt_pt+post_TS] = 8
    data.index = data['Date_time'] 
    return(data)
  

#set processing variables
T_avg = '10s'  #Time averaging
ref_cts_diff_limit = 1e5 #filter to cut data where we lost the peak based on ref cell cts diff
ref_delta_value = 1000 #arbitary number for ref cell normalisation
pre_taskswitch = 1   #number of seconds before a task switch to be ignored
post_taskswitch = 5   #number of seconds after a task switch to be ignored
pre_peakfind = 1   #number of seconds before ref_cts_diff drops below ref_cts_diff_limit to be ignored
post_peakfind = 2   #number of seconds after ref_cts_diff drops below ref_cts_diff_limit to be ignored
cal_max_conc = 5000   #maximum cal SO2 concentration value to be used in calculating sensitivity
Diagnostic_plot_lst = ['sig_A_diff_cts', 'sig_B_diff_cts', 'ref_cts_diff', 'Task_y', 'Laser_Power_PT_0', 'PC_Pressure'] #list of variables to be plotted as diagnostics

#load in the processed sig and ref data and then the hk and time average to 1s data
sig_cts_data = lif.load_counts(r"Processed_20250530")
print('Loading signal cell data between:')
print(sig_cts_data.index[0])
print(sig_cts_data.index[-1])
#sig_cts_data = sig_cts_data.add_suffix('_sig')
Avg_1s_sig = sig_cts_data.resample('1s').mean()

# ref_cts_data = lif.load_counts(r"processed_data_ref")
# print('Loading ref cell data between:')
# print(ref_cts_data.index[0])
# print(ref_cts_data.index[-1])
# ref_cts_data = ref_cts_data.add_suffix('_ref')
# Avg_1s_ref = ref_cts_data.resample('1s').mean()

hk_data = lif.load_hk(r"LIFHK_20250530", no_cylinder_conc=5000)
print('Loading hk data between:')
print(hk_data.index[0])
print(hk_data.index[-1])
Avg_1s_hk = hk_data.resample('1s').mean()

#merge above data into a single 1s dataframe
Avg_1s_data = pd.merge(Avg_1s_sig, Avg_1s_hk, left_index=True, right_index=True)

#filtering for change of task and ref cell diff signal
Avg_1s_data = set_flags(Avg_1s_data, pre_taskswitch, post_taskswitch, pre_peakfind, post_peakfind, ref_cts_diff_limit)

#Plot diagnostics
Diagnostic_plots(Avg_1s_data, Diagnostic_plot_lst)

#Ref cell signal normilisation
Avg_1s_data['Sig_A_ctsdiff_ref_norm'] = Avg_1s_data['sig_A_diff_cts']/Avg_1s_data['ref_cts_diff']*ref_delta_value
Avg_1s_data['Sig_B_ctsdiff_ref_norm'] = Avg_1s_data['sig_B_diff_cts']/Avg_1s_data['ref_cts_diff']*ref_delta_value

#analyse cals (can switch plotting on or off) and calculate and apply ref normalised and non ref normalised sensitivties
#Currently sensitivities are just an average of the cals with R2 > 0.98, need to expand this for interpolation etc. 
Std_cal_summary_A, Refnorm_cal_summary_A = Analyse_cals(Avg_1s_data, plot = True, max_conc = cal_max_conc, cell = 'A')
Std_C_A = Std_cal_summary_A['Slope'].where(Std_cal_summary_A['R2'] > 0.9).mean()
Ref_norm_C_A = Refnorm_cal_summary_A['Slope'].where(Refnorm_cal_summary_A['R2'] > 0.9).mean()
Avg_1s_data['Cald_NO_A_ppt_ref_norm'] = Avg_1s_data['Sig_A_ctsdiff_ref_norm']/Ref_norm_C_A
Avg_1s_data['Cald_NO_A_ppt'] = Avg_1s_data['sig_A_diff_cts']/Std_C_A

Std_cal_summary_B, Refnorm_cal_summary_B = Analyse_cals(Avg_1s_data, plot = True, max_conc = cal_max_conc, cell = 'B')
Std_C_B = Std_cal_summary_B['Slope'].where(Std_cal_summary_B['R2'] > 0.9).mean()
Ref_norm_C_B = Refnorm_cal_summary_B['Slope'].where(Refnorm_cal_summary_B['R2'] > 0.9).mean()
Avg_1s_data['Cald_NO_B_ppt_ref_norm'] = Avg_1s_data['Sig_B_ctsdiff_ref_norm']/Ref_norm_C_B
Avg_1s_data['Cald_NO_B_ppt'] = Avg_1s_data['sig_B_diff_cts']/Std_C_B

#Filter data based on task
Avg_1s_data['Ambient_NO_A_ppt_ref_norm'] = Avg_1s_data['Cald_NO_A_ppt_ref_norm'].where((Avg_1s_data['Task_y'] == 0) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Zero_A_ppt_ref_norm'] = Avg_1s_data['Cald_NO_A_ppt_ref_norm'].where((Avg_1s_data['Task_y'] == 4) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Ambient_NO_A_ppt'] = Avg_1s_data['Cald_NO_A_ppt'].where((Avg_1s_data['Task_y'] == 0) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Zero_A_ppt'] = Avg_1s_data['Cald_NO_A_ppt'].where((Avg_1s_data['Task_y'] == 4) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))

Avg_1s_data['Ambient_NO_B_ppt_ref_norm'] = Avg_1s_data['Cald_NO_B_ppt_ref_norm'].where((Avg_1s_data['Task_y'] == 0) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Zero_B_ppt_ref_norm'] = Avg_1s_data['Cald_NO_B_ppt_ref_norm'].where((Avg_1s_data['Task_y'] == 4) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Ambient_NO_B_ppt'] = Avg_1s_data['Cald_NO_B_ppt'].where((Avg_1s_data['Task_y'] == 0) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))
Avg_1s_data['Zero_B_ppt'] = Avg_1s_data['Cald_NO_B_ppt'].where((Avg_1s_data['Task_y'] == 4) & (Avg_1s_data['Peak_find_flag'] == 0) & (Avg_1s_data['ref_cts_diff'] > ref_cts_diff_limit))

#time average processed data
T_avg_data = Avg_1s_data.resample(T_avg).mean()

#plots
plt.plot(T_avg_data.index, T_avg_data['Ambient_NO_A_ppt_ref_norm'], color = 'red', label = 'Cell A')
plt.plot(T_avg_data.index, T_avg_data['Ambient_NO_B_ppt_ref_norm'], color = 'blue', label = 'Cell B')
plt.title('Ambient data '+T_avg+' average')
plt.legend()
plt.show()

plt.plot(T_avg_data.index, T_avg_data['Zero_A_ppt_ref_norm'], color = 'red', label = 'Cell A')
plt.plot(T_avg_data.index, T_avg_data['Zero_B_ppt_ref_norm'], color = 'blue', label = 'Cell B')
plt.title('Zero data '+T_avg+' average')
plt.legend()
plt.show()

T_avg_data.to_csv(str(T_avg)+'_Avg_data.csv')