
diag_plots_dict = {
    0: {
        'name': 'temperatures'
        , 'x_header': 'Time_s'
        , 'y_header': ['T_board_LIF_cell', 'Thermistor_7']
        , 'x_label': 'time UTC'
        , 'y_label': 'temperature / degC * 100'
    },
    1: {
        'name': 'flows'
        , 'x_header': 'Time_s'
        , 'y_header': ['Sig_Cell_Flow', 'Ref_Cell_Flow']
        , 'x_label': 'time UTC'
        , 'y_label': 'flow / slpm'
    }
}

html_prefs = lif_utils.load_html_prefs()

if gen_diag_plots == True:

    for plot_key in list(html_prefs):

        if '[' and ']' in html_prefs[plot_key]['y_header']:
            y_arr = {header: HK_data[header] for header in
                     html_prefs[plot_key]['y_header'].replace('[', '').replace(']', '').split(sep=',')}

        else:
            y_arr = HK_data[html_prefs[plot_key]['y_header']]

        utils.gen_HTML_plots(HK_data[html_prefs[plot_key]['x_header']]
                             , y_arr, html_prefs[plot_key]['name'])